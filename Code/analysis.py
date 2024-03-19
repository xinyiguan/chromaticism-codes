import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Self, Literal, Optional, Tuple

import pandas as pd
import pingouin as pg


def correlation(series1: pd.Series, series2: pd.Series):
    return pg.corr(x=series1, y=series2)


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


def get_corpus_df(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    result = df[df["corpus"] == corpus]
    return result


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


def get_copora_list_in_period_table(df: pd.DataFrame):
    renaissance, baroque, classical, early_romantic, late_romantic = corpora_in_periods_dfs(df=df)

    renaissance_corpus = [x.replace("_", " ") for x in renaissance['corpus'].unique()]
    baroque_corpus = [x.replace("_", " ") for x in baroque['corpus'].unique()]
    classical_corpus = [x.replace("_", " ") for x in classical['corpus'].unique()]
    early_romantic_corpus = [x.replace("_", " ") for x in early_romantic['corpus'].unique()]
    late_romantic_corpus = [x.replace("_", " ") for x in late_romantic['corpus'].unique()]

    data = {
        'Renaissance': renaissance_corpus,
        'Baroque': baroque_corpus,
        'Classical': classical_corpus,
        'Early Romantic': early_romantic_corpus,
        'Late Romantic': late_romantic_corpus
    }

    # Start building the LaTeX table
    latex_table = "\\begin{tabular}{ll}\n\\toprule\nPeriod & Corpus \\\\\n\\midrule\n"
    total_periods = len(data)
    current_period_count = 0

    for period, corpus_list in data.items():
        current_period_count += 1
        for i, corpus in enumerate(corpus_list):
            if i == 0:
                latex_table += f"{period} & {corpus} \\\\\n"
            else:
                latex_table += f" & {corpus} \\\\\n"

        if current_period_count < total_periods:  # add horizontal line except for the last period
            latex_table += "\\midrule\n"

    # end the LaTeX table
    latex_table += "\\bottomrule\n\\end{tabular}"

    latex_table_path = os.path.join("results_latex", 'corpora_by_period.txt')
    with open(latex_table_path, 'w') as file:
        file.write(latex_table)

    return latex_table

@dataclass
class MusicExamples:
    beethoven_17_02: pd.DataFrame
    bach_bwv808: pd.DataFrame
    mozart_k331: pd.DataFrame
    chopin_06_02: pd.DataFrame

    @classmethod
    def dfs(cls, path: str = "data/chord_indices.tsv") -> Self:
        df = pd.read_csv(path, sep="\t")
        df['quarterbeats'] = df['quarterbeats'].apply(lambda x: float(Fraction(x)) if '/' in x else float(x))

        beethoven = df[(df["corpus"] == "beethoven_piano_sonatas") & (df["piece"] == "17-3")]

        bach = df[(df['corpus'] == 'bach_en_fr_suites') & (df['piece'] == 'BWV808_04_Sarabande')]

        mozart = df[(df['corpus'] == 'mozart_piano_sonatas') & (df['piece'] == 'K331-1')]

        chopin = df[(df["corpus"] == "chopin_mazurkas") & (df["piece"] == "BI60-2op06-2")]

        instance = cls(beethoven_17_02=beethoven,
                       bach_bwv808=bach,
                       mozart_k331=mozart,
                       chopin_06_02=chopin)
        return instance

    def get_example_segment(self, composer: Literal["beethoven", "bach", "mozart", "chopin"],
                            version: Optional[
                                Literal["thema", "var1", "var2", "var3", "var4", "var5", "var6"]]) -> pd.DataFrame:
        if composer == "beethoven":
            assert version is None
            result = self.beethoven_17_02[self.beethoven_17_02['quarterbeats'].between(0, 135 / 4)]

        elif composer == "bach":
            assert version is None
            result = self.bach_bwv808[self.bach_bwv808['quarterbeats'].between(0, 22)]

        elif composer == "chopin":
            assert version is None
            result = self.chopin_06_02[self.chopin_06_02['quarterbeats'].between(24, 45)]

        elif composer == "mozart":
            mozart = self.mozart_k331
            if version == "thema":
                result = mozart[(mozart["quarterbeats"].between(0, 45 / 2))]

            elif version == "var1":
                result = mozart[(mozart["quarterbeats"].between(111 / 2, 153 / 2))]

            elif version == "var2":
                result = mozart[(mozart["quarterbeats"].between(109, 261 / 2))]

            elif version == "var3":
                result = mozart[(mozart["quarterbeats"].between(162, 369 / 2))]

            elif version == "var4":
                result = mozart[(mozart["quarterbeats"].between(216, 477 / 2))]

            elif version == "var5":
                result = mozart[(mozart["quarterbeats"].between(1085 / 4, 293))]

            elif version == "var6":
                result = mozart[(mozart["quarterbeats"].between(325, 354))]

            else:
                raise ValueError

        else:
            raise ValueError
        result["version"] = f'{version}'
        result = result[["version", "globalkey", "localkey", "chord", "root", "RC", "ct", "CTC", "nct", "NCTC"]]
        return result

    def save_latex_table(self):
        raise NotImplementedError

    def get_GlobalIndices_table(self, composer: Literal["beethoven", "bach", "mozart", "chopin"]) -> pd.DataFrame:
        if composer == "mozart":
            thema = self.get_example_segment(composer="mozart", version="thema")
            var1 = self.get_example_segment(composer="mozart", version="var1")
            var2 = self.get_example_segment(composer="mozart", version="var2")
            var3 = self.get_example_segment(composer="mozart", version="var3")
            var4 = self.get_example_segment(composer="mozart", version="var4")
            var5 = self.get_example_segment(composer="mozart", version="var5")
            var6 = self.get_example_segment(composer="mozart", version="var6")

            dfs = []
            for x in [thema, var1, var2, var3, var4, var5, var6]:
                data = [x["version"].unique()[0], x["RC"].mean(), x["CTC"].mean(), x["NCTC"].mean()]
                cols = ["version", "RC", "CTC", "NCTC"]
                result = pd.DataFrame([data], columns=cols)
                dfs.append(result)

            results = pd.concat(dfs)

        else:
            df = self.get_example_segment(composer=composer, version=None)
            data = [df["version"].unique()[0], df["RC"].mean(), df["CTC"].mean(), df["NCTC"].mean()]
            cols = ["version", "RC", "CTC", "NCTC"]
            results = pd.DataFrame([data], columns=cols)
        return results


if __name__ == "__main__":
    a = MusicExamples.dfs()
    beethoven = a.get_GlobalIndices_table(composer="beethoven")
    bach = a.get_GlobalIndices_table(composer="bach")
    mozart = a.get_GlobalIndices_table(composer="mozart")
    chopin = a.get_GlobalIndices_table(composer="chopin")
    with pd.option_context('display.max_columns', None):
        print(f'{beethoven=}')
        print(f'{bach=}')
        print(f'{mozart=}')
        print(f'{chopin=}')