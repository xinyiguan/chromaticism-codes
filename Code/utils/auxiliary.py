# auxiliary functions for preprocessing, analysis
import os
from typing import Literal, Tuple

import pandas as pd


# preprocessing/computing metrics ___________ :


def determine_period_Johannes(row):
    y = row["piece_year"]
    if y < 1650:
        p = "pre-Baroque"
    elif 1650 <= y < 1750:
        p = "Baroque"
    elif 1750 <= y < 1800:
        p = "Classical"
    elif y >= 1800:
        p = "Extended tonality"
    else:
        raise ValueError
    return p


def determine_period(row):
    y = row["piece_year"]
    if y < 1662:
        p = "Renaissance"
    elif 1662 <= y < 1763:
        p = "Baroque"
    elif 1763 <= y < 1821:
        p = "Classical"
    elif 1821 <= y < 1869:
        p = "Early Romantic"
    elif y >= 1869:
        p = "Late Romantic"
    else:
        raise ValueError
    return p


def determine_period_id(row):
    if 'period_Johannes' in row.index:
        if row["period_Johannes"] == "pre-Baroque":
            id = 1
        elif row["period_Johannes"] == "Baroque":
            id = 2
        elif row["period_Johannes"] == "Classical":
            id = 3
        elif row["period_Johannes"] == "Extended tonality":
            id = 4
        else:
            raise ValueError
    elif 'period' in row.columns:
        if row["period"] == "Renaissance":
            id = 1
        elif row["period"] == "Baroque":
            id = 2
        elif row["period"] == "classical":
            id = 3
        elif row["period"] == "Early Romantic":
            id = 4
        elif row["period"] == "Late Romantic":
            id = 5
        else:
            raise ValueError
    else:
        raise ValueError

    return id


# ___________
def get_period_df_Johannes(df: pd.DataFrame, period: Literal[
    "pre-Baroque", "Baroque", "Classical", "Extended tonality"]) -> pd.DataFrame:
    t1, t2, t3 = (1650, 1750, 1800)

    pre_Baroque = df[df["piece_year"] < t1]
    Baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
    Classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
    extended_tonality = df[df["piece_year"] >= t3]

    if period == "pre-Baroque":
        return pre_Baroque
    elif period == "Baroque":
        return Baroque
    elif period == "Classical":
        return Classical
    elif period == "Extended tonality":
        return extended_tonality
    else:
        raise ValueError


def get_period_df(df: pd.DataFrame, period: Literal[
    "Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]) -> pd.DataFrame:
    t1, t2, t3, t4 = (1662, 1763, 1821, 1869)

    late_renaissance = df[df["piece_year"] < t1]
    baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
    classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
    early_romantic = df[(t3 <= df["piece_year"]) & (df["piece_year"] < t4)]
    late_romantic = df[df["piece_year"] >= t4]

    if period == "Renaissance":
        return late_renaissance
    elif period == "Baroque":
        return baroque
    elif period == "Classical":
        return classical
    elif period == "Early Romantic":
        return early_romantic
    elif period == "Late Romantic":
        return late_romantic
    else:
        raise ValueError


def corpora_in_periods_dfs(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df[["corpus", "piece", "piece_year"]]
    df = df.reset_index(drop=True)

    renaissance = get_period_df(df=df, period="Renaissance")
    renaissance.loc[:, "period"] = "Renaissance"
    # renaissance["period"] = "Renaissance"

    baroque = get_period_df(df=df, period="Baroque")
    baroque.loc[:, "period"] = "Baroque"

    classical = get_period_df(df=df, period="Classical")
    classical.loc[:, "period"] = "Classical"

    early_romantic = get_period_df(df=df, period="Early Romantic")
    early_romantic.loc[:, "period"] = "Early Romantic"

    late_romantic = get_period_df(df=df, period="Late Romantic")
    late_romantic.loc[:, "period"] = "Late Romantic"

    # corpora_by_period_df = pd.concat([renaissance, baroque, classical, early_romantic, late_romantic])
    # corpora_by_period_df = corpora_by_period_df.set_index("period")

    return renaissance, baroque, classical, early_romantic, late_romantic


def create_results_folder(analysis_name: str, repo_dir: str):
    dir = f"{repo_dir}/Results/{analysis_name}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
