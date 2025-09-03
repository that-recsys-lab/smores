from lenskit.data.items import ItemList
import pandas as pd
import numpy as np
from icecream import ic

def itemList2rankedTuples(item_list: ItemList):
    ids = item_list.ids()
    scores = item_list.scores()
    ranks = item_list.ranks()

    sort_indices = np.argsort(ranks)

    return [(ids[i], scores[i]) for i in sort_indices]