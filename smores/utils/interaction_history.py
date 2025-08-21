import pyarrow as pa
from icecream import ic

from lenskit.data import DatasetBuilder, Dataset, ItemList

class InteractionHistory:
    INTERACTION_COLUMNS = ['user_id', 'item_id', 'rating', 'time']

    def __init__(self):
        self.interaction_table: pa.Table = None

    # Single interaction is not efficient
    def add_interaction(self, user_id, item_id, interaction, time):
        column_lists = [[user_id], [item_id], [interaction], [time]]
        batch = self.column_lists_to_batch(column_lists)
        table1 = pa.Table.from_batches([batch])
        if self.interaction_table is None:
            self.interaction_table = table1
        else:
            new_table = pa.concat_tables([self.interaction_table, table1])
            self.interaction_table = new_table

    
    def add_interactions(self, interaction_list):
        column_lists = zip(*interaction_list)
        batch = self.column_lists_to_batch(column_lists)
        batch_table = pa.Table.from_batches([batch])
        if self.interaction_table is None:
            self.interaction_table = batch_table
        else:
            new_table = pa.concat_tables([self.interaction_table, batch_table])
            self.interaction_table = new_table

    # TODO: What is the table format? Is an item list OK?
    def add_interactions_itemlist(self, consumer_id: int, interaction_items: ItemList):
        interaction_table = interaction_items.to_arrow()
        interaction_table = interaction_table.add_column(0, self.INTERACTION_COLUMNS[0], [[consumer_id]*len(interaction_items)])
        if self.interaction_table is None:
            self.interaction_table = interaction_table
        else:
            new_table = pa.concat_tables(self.interaction_table, interaction_table)
            self.interaction_table = new_table
    

    def to_dataset(self, old_dataset: (Dataset|None) = None) -> Dataset:
        if old_dataset is None:
            builder = DatasetBuilder(None)
        else:
            builder = DatasetBuilder(old_dataset)

        builder.add_interactions('interaction', self.interaction_table, 
                                 entities=[self.INTERACTION_COLUMNS[0], self.INTERACTION_COLUMNS[1]],
                                 missing='insert', allow_repeats=False, default=True)
        return builder.build()
    
    def lists_to_arrays(self, lists):
        return [pa.array(lst) for lst in lists]
    
    def column_lists_to_batch(self, column_lists):
        column_arrays = self.lists_to_arrays(column_lists)
        batch = pa.record_batch(column_arrays, names=InteractionHistory.INTERACTION_COLUMNS)
        return batch



    
    

