from fractions import Fraction
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


def get_bwv808_example_CI(version: Literal["original", "agrements"],
                          tsv_path: str = "data/chord_indices.tsv") -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["corpus", "piece", "quarterbeats", "chord", "chord_tones", "tones_in_span", "root",
                              "r_chromaticity", "ct", "ct_chromaticity", "nct", "nct_chromaticity"])
    df['quarterbeats'] = df['quarterbeats'].apply(lambda x: float(Fraction(x)) if '/' in x else float(x))

    if version == "original":
        bwv808 = df[(df['corpus'] == 'bach_en_fr_suites') & (df['piece'] == 'BWV808_04_Sarabande') & (
            df['quarterbeats'].between(0, 22))]

    else:
        bwv808 = df[(df['corpus'] == 'bach_en_fr_suites') & (df['piece'] == 'BWV808_04a_Agrements_de_la_Sarabande') & (
            df['quarterbeats'].between(0, 22))]

    # with pd.option_context('display.max_columns', None):
    #     print(bwv808)

    data = [bwv808["r_chromaticity"].mean(), bwv808["ct_chromaticity"].mean(), bwv808["nct_chromaticity"].mean()]

    cols = ["RC", "CTC", "NCTC"]

    results = pd.DataFrame([data], columns=cols)

    results.index = [f'{version}']
    return results


def get_k331_1_example_CI(version: Literal["thema", "var2", "var5", "var6"],
                          tsv_path: str = "data/chord_indices.tsv") -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["corpus", "piece", "quarterbeats", "chord", "chord_tones", "tones_in_span", "root",
                              "r_chromaticity", "ct", "ct_chromaticity", "nct", "nct_chromaticity"])
    df['quarterbeats'] = df['quarterbeats'].apply(lambda x: float(Fraction(x)) if '/' in x else float(x))

    mozart = df[(df['corpus'] == 'mozart_piano_sonatas') & (df['piece'] == 'K331-1')]

    if version == "thema":
        k331 = mozart[(mozart["quarterbeats"].between(0, 45 / 2))]

    elif version == "var2":
        k331 = mozart[(mozart["quarterbeats"].between(109, 261 / 2))]

    elif version == "var5":
        k331 = mozart[(mozart["quarterbeats"].between(1085 / 4, 293))]

    elif version == "var6":
        k331 = mozart[(mozart["quarterbeats"].between(325, 354))]

    else:
        raise ValueError

    k331 = k331[
        ["chord", "root", "r_chromaticity", "ct", "ct_chromaticity", "nct", "nct_chromaticity"]]

    k331["version"] = f'{version}'

    with pd.option_context('display.max_columns', None):
        k331.to_csv(f"k331_{version}.tsv", sep="\t")

    data = [k331["r_chromaticity"].mean(), k331["ct_chromaticity"].mean(), k331["nct_chromaticity"].mean()]

    cols = ["RC", "CTC", "NCTC"]

    results = pd.DataFrame([data], columns=cols)

    results.index = [f'{version}']
    return results

    # return k331


def get_k331_CI_table():
    thema = get_k331_1_example_CI(version="thema")
    var5 = get_k331_1_example_CI(version="var5")
    var6 = get_k331_1_example_CI(version="var6")

    df = pd.concat([thema, var5, var6])

    latex = df.to_latex()


    # with pd.option_context('display.max_columns', None):
    #     print(df)
    print(latex)

if __name__ == "__main__":
    # result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    # get_copora_list_in_period(df=result_df)

    # original = get_bwv808_example_CI(version="original")
    # print(f'{original}')
    #
    # agrements = get_bwv808_example_CI(version="agrements")
    # print(f'{agrements}')

    get_k331_CI_table()