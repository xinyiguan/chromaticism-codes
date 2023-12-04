import os
from dataclasses import dataclass
from typing import Tuple, Literal, Self

import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
from analysis import MusicExamples
import seaborn as sns


def piece_distribution_with_EraDivision(df: pd.DataFrame,
                                        show_fig: bool = False,
                                        save_histogram: bool = False) -> Tuple[int, int, int, int]:
    # global variables
    DPI = 300

    h = sns.histplot(df["piece_year"], kde=True, stat="probability", bins=40,
                     kde_kws={'bw_adjust': 0.6})
    h.set_xlabel("Year", fontsize=15)
    h.set_ylabel("probability", fontsize=15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Get the data from the plot
    lines = plt.gca().get_lines()
    xs, ys = lines[0].get_data()

    # Find the local minima in the histogram
    mininds = []
    a, b = -1, -1
    for i, c in enumerate(ys):
        if a > b and b < c:
            mininds.append(i)
        a, b = b, c

    # Extract the x-values of the minima
    t1, _, t2, t3, t4 = [xs[i] for i in mininds]

    # Add vertical lines at the minima
    for b in [t1, t2, t3, t4]:
        h.axvline(b, c="gray", ls="--", zorder=-2)

    t1 = round(t1)
    t2 = round(t2)
    t3 = round(t3)
    t4 = round(t4)

    print(f'{t1=}, {t2=}, {t3=}, {t4=} ')

    if show_fig:
        plt.show()
    if save_histogram:
        p = h.get_figure()
        p.savefig("figs/Figure_histogram.pdf", dpi=DPI)
    return t1, t2, t3, t4


@dataclass
class PlotBeethovenExample:
    df: pd.DataFrame

    @classmethod
    def beethoven(cls) -> Self:
        exs = MusicExamples.dfs()
        b = exs.get_example_segment(composer="beethoven", version=None)
        b = b.reset_index()
        instance = cls(df=b)
        return instance

    def beethoven_chordCI_table(self):
        cols = ["chord", "root", "RC", "ct", "CTC", "nct", "NCTC"]
        cleaned_df = self.df[cols]
        cleaned_df = cleaned_df.set_index('chord')
        transposed_df = cleaned_df.transpose()
        latex = transposed_df.to_latex()

        latex_table_path = os.path.join("latex_tables", 'beethoven_chordCI.txt')
        with open(latex_table_path, 'w') as file:
            file.write(latex)
        return latex

    def beethoven_chordCI_lineplot(self):
        markerline, stemlines, baseline = plt.stem(self.df["index"], self.df["RC"], '-.', label='RC')
        markerline, stemlines, baseline = plt.stem(self.df["index"], self.df["CTC"], '-.', label='CTC')
        markerline, stemlines, baseline = plt.stem(self.df["index"], self.df["NCTC"], '-.', label='NCTC')
        plt.setp(baseline, 'color', 'r', 'linewidth', 2)

        plt.show()


if __name__ == "__main__":
    a = PlotBeethovenExample.beethoven()
    a.beethoven_chordCI_lineplot()
