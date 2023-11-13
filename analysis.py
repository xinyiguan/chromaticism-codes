from typing import Literal, Tuple

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


def add_periods_to_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def get_copora_list_in_period(df: pd.DataFrame):
    renaissance, baroque, classical, early_romantic, late_romantic = add_periods_to_df(df=df)

    renaissance_corpus = renaissance['corpus'].unique()
    renaissance_corpus = [x.replace("_", " ") for x in renaissance_corpus]

    baroque_corpus = baroque['corpus'].unique()
    baroque_corpus = [x.replace("_", " ") for x in baroque_corpus]

    classical_corpus = classical['corpus'].unique()
    classical_corpus = [x.replace("_", " ") for x in classical_corpus]

    early_romantic_corpus = early_romantic['corpus'].unique()
    early_romantic_corpus = [x.replace("_", " ") for x in early_romantic_corpus]

    late_romantic_corpus = late_romantic['corpus'].unique()
    late_romantic_corpus = [x.replace("_", " ") for x in late_romantic_corpus]

    data = {
        'Renaissance': renaissance_corpus,
        'Baroque': baroque_corpus,
        'Classical': classical_corpus,
        'Early Romantic': early_romantic_corpus,
        'Late Romantic': late_romantic_corpus
    }
    # Start building the LaTeX table
    latex_table = "\\begin{tabular}{ll}\n\\toprule\nPeriod & Corpus \\\\\n\\midrule\n"

    for period, corpus_list in data.items():
        for i, corpus in enumerate(corpus_list):
            if i == 0:
                latex_table += f"{period} & {corpus} \\\\\n"
            else:
                latex_table += f" & {corpus} \\\\\n"

    # End the LaTeX table
    latex_table += "\\bottomrule\n\\end{tabular}"

    print(latex_table)
    return latex_table


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    get_copora_list_in_period(df=result_df)
