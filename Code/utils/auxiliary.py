# auxiliary functions for preprocessing, analysis
import os
from typing import Literal, Tuple

import pandas as pd


# preprocessing/computing metrics ___________ :


def determine_period_Johannes(row):
    y = row["piece_year"]
    if y < 1650:
        p = "pre_Baroque"
    elif 1650 <= y < 1750:
        p = "Baroque"
    elif 1750 <= y < 1800:
        p = "Classical"
    elif y >= 1800:
        p = "extended_tonality"
    else:
        raise ValueError
    return p


def determine_period(row):
    y = row["piece_year"]
    if y < 1662:
        p = "renaissance"
    elif 1662 <= y < 1763:
        p = "baroque"
    elif 1763 <= y < 1821:
        p = "classical"
    elif 1821 <= y < 1869:
        p = "early_romantic"
    elif y >= 1869:
        p = "late_romantic"
    else:
        raise ValueError
    return p

def determine_period_id(row):
    if 'period_Johannes' in row.index:
        if row["period_Johannes"] == "pre_Baroque":
            id=1
        elif row["period_Johannes"] == "Baroque":
            id=2
        elif row["period_Johannes"] == "Classical":
            id=3
        elif row["period_Johannes"] == "extended_tonality":
            id=4
        else:
            raise ValueError
    elif 'period' in row.columns:
        if row["period"] == "renaissance":
            id=1
        elif row["period"] == "baroque":
            id=2
        elif row["period"] == "classical":
            id=3
        elif row["period"] == "early_romantic":
            id=4
        elif row["period"] == "late_romantic":
            id=5
        else:
            raise ValueError
    else:
        raise ValueError

    return id


# ___________
def get_period_df_Johannes(df: pd.DataFrame, period: Literal[
    "pre_Baroque", "Baroque", "Classical", "extended_tonality"]) -> pd.DataFrame:
    t1, t2, t3 = (1650, 1750, 1800)

    pre_Baroque = df[df["piece_year"] < t1]
    Baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
    Classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
    extended_tonality = df[df["piece_year"] >= t3]

    if period == "pre_Baroque":
        return pre_Baroque
    elif period == "Baroque":
        return Baroque
    elif period == "Classical":
        return Classical
    elif period == "extended_tonality":
        return extended_tonality
    else:
        raise ValueError


def get_period_df(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]) -> pd.DataFrame:
    t1, t2, t3, t4 = (1662, 1763, 1821, 1869)

    late_renaissance = df[df["piece_year"] < t1]
    baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
    classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
    early_romantic = df[(t3 <= df["piece_year"]) & (df["piece_year"] < t4)]
    late_romantic = df[df["piece_year"] >= t4]

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

def corpora_in_periods_dfs(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df[["corpus", "piece", "piece_year"]]
    df = df.reset_index(drop=True)

    renaissance = get_period_df(df=df, period="renaissance")
    renaissance.loc[:, "period"] = "Renaissance"
    # renaissance["period"] = "Renaissance"

    baroque = get_period_df(df=df, period="baroque")
    baroque.loc[:, "period"] = "Baroque"

    classical = get_period_df(df=df, period="classical")
    classical.loc[:, "period"] = "Classical"

    early_romantic = get_period_df(df=df, period="early_romantic")
    early_romantic.loc[:, "period"] = "Early Romantic"

    late_romantic = get_period_df(df=df, period="late_romantic")
    late_romantic.loc[:, "period"] = "Late Romantic"

    # corpora_by_period_df = pd.concat([renaissance, baroque, classical, early_romantic, late_romantic])
    # corpora_by_period_df = corpora_by_period_df.set_index("period")

    return renaissance, baroque, classical, early_romantic, late_romantic


def create_results_folder(analysis_name: str, repo_dir: str):


    dir = f"{repo_dir}/Results/{analysis_name}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

