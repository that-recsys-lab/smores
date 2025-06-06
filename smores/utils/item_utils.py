from lenskit.data.items import ItemList
import pandas as pd
from icecream import ic

def itemList2rankedTuples(item_list: ItemList):
    df = item_list.to_df(ids=True)
    df_sorted = df.sort_values(by=['rank'])
    return [(row[0], row[1]) for row in df_sorted.itertuples(index=False, name=None)]
