from lenskit.data.items import ItemList
import pandas as pd
import numpy as np
from icecream import ic

def itemList2rankedTuples(item_list: ItemList):
    ids = item_list.ids()
    scores = item_list.scores()
    ranks = item_list.ranks()

    sort_indices = np.argsort(ranks)

    if scores is None:
        return [(ids[i], 1) for i in sort_indices]

    return [(ids[i], scores[i]) for i in sort_indices]
