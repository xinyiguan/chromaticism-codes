from typing import Literal

import pandas as pd


# generic scattered plots for indices

def scatter_plot(df: pd.DataFrame, index: Literal["WLC", "OLC", "WLD"]):
    raise NotImplementedError


