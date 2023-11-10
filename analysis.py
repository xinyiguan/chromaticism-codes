from typing import Literal

import pandas as pd
import pingouin as pg


def correlation(series1: pd.Series, series2: pd.Series):
    return pg.corr(x=series1, y=series2)


def get_period_df(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]) -> pd.DataFrame:
    late_renaissance = df[df["piece_year"] < 1662]
    baroque = df[(1662 <= df["piece_year"]) & (df["piece_year"] < 1761)]
    classical = df[(1761 <= df["piece_year"]) & (df["piece_year"] < 1820)]
    early_romantic = df[(1820 <= df["piece_year"]) & (df["piece_year"] < 1871)]
    late_romantic = df[df["piece_year"] >= 1871]

    if period == "renaissance":
        return late_renaissance
    elif period == "baroque":
        return baroque
    elif period == "classical":
        return classical
    elif period == "early_romantic":
        return early_romantic
    elif period == "late_romantic":
        return late_romantic
    else:
        raise ValueError


if __name__=="__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    renaissance = get_period_df(result_df, period='renaissance')
    print(renaissance)