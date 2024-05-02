# script slightly adapted from Edward
import os
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# def dissonance(x, weights):
#     """
#     Calculate dissonance of `x`, a list of interval classes, given weights for classes 1--6
#     """
#     pmf = np.array([x.count(i + 1) for i in range(6)]) / len(x)
#     res = np.nansum([pmf[i] * weights[i] for i in range(6)])
#     return res

def dissonance(x, weights):
    """
    Calculate dissonance of `x`, a list of interval classes, given weights for classes 1--6
    """
    pmf = np.array([x.count(i + 1) for i in range(6)])
    res = np.nansum([pmf[i] * weights[i] for i in range(6)])
    return res


def test_weights(interval_classes: List, ratings, weights):
    """
    Calculate correlation between dissonance scores and ratings
    """
    n = len(interval_classes)
    if n != len(ratings):
        raise Exception("`interval_classes` and `ratings` must be the same length")

    res = np.zeros(n)
    for i in range(n):
        res[i] = dissonance(interval_classes[i], weights)

    cor, pvalue = pearsonr(res, ratings)

    return cor, pvalue


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'

    # Import behavioural data
    # (invert rating scale)
    beh_data_path = f'{repo}Data/bowling2018/bowling2018_consonance.tsv'
    data = pd.read_table(beh_data_path)
    data.interval_class = [[int(y) for y in x.split(",")] for x in data.interval_class]
    data.rating = 4 - data.rating

    # Correlation when using weights [6, 4, 3, 2, 1, 5] (scaled between 0 and 1)
    print(
        "ranks 1--6:",
        test_weights(data.interval_class, data.rating, [1.0, 0.6, 0.4, 0.2, 0.0, 0.8])
    )
