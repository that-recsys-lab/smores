import pyarrow as pa
from lenskit.data import DatasetBuilder, Dataset

class InteractionHistory:
    def __init__(self):
        self.interaction_table: pa.Table = None

    # Single interaction is not efficient
    def add_interaction(self, user_id, item_id, interaction, time):
        inter_array = pa.Array(user_id, item_id, interaction, time)
        batch = pa.RecordBatch.from_arrays([inter_array])
        table1 = pa.Table.from_batches(batch)
        if self.interaction_table is None:
            self.interaction_table = table1
        else:
            new_table = pa.concat_tables([self.interaction_table, table1])
            self.interaction_table = new_table

    
    def add_interactions(self, interaction_list):
        arrays = [pa.Array(user_id, item_id, interaction, time) 
                  for user_id, item_id, interaction, time in interaction_list]
        batch = pa.RecordBatch.from_arrays(arrays)
        batch_table = pa.Table.from_batches(batch)
        if self.interaction_table is None:
            self.interaction_table = batch_table
        else:
            new_table = pa.concat_tables([self.interaction_table, batch_table])
            self.interaction_table = new_table
    

    def to_dataset(self) -> Dataset:
        builder = DatasetBuilder(self.interaction_table)
        return builder.build()
    
    

