from typing import Literal

import pandas as pd
import seaborn as sns
import statsmodels as sm
from matplotlib import pyplot as plt
from analysis import get_period_df


def piece_distribution_histogram(tsv_path: str = "data/piece_chromaticities.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")

    # global variables
    DPI = 300
    fs = 30

    # Create the histogram plot
    sns.histplot(df["piece_year"], kde=True, stat="probability", bins=40,
                 kde_kws={'bw_adjust': 0.6})

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
        plt.axvline(b, c="gray", ls="--", zorder=-2)

    print(f'{t1=}, {t2=}, {t3=}, {t4=} ')

    # Set labels and show the plot
    plt.xlabel("year")
    plt.ylabel("probability")
    plt.savefig("figs/Figure1_histogram.pdf", dpi=DPI)

    plt.show()


def plot_chromaticity_correlation(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]):
    chorsen_df = get_period_df(df=df, period=period)

    ax1 = plt.subplot(221)
    ax1 = sns.scatterplot(x=chorsen_df["mean_r_chromaticity"].tolist(),
                          y=chorsen_df["mean_ct_chromaticity"].tolist(),
                          s=8, c=chorsen_df["corpus_id"], cmap="Purples")
    ax1.set_xlabel('RC')
    ax1.set_ylabel('CTC')

    # ax1.set_title("r-ct")

    ax2 = plt.subplot(222)
    ax2 = sns.scatterplot(x=chorsen_df["mean_r_chromaticity"].tolist(),
                          y=chorsen_df["mean_nct_chromaticity"].tolist(),
                          s=5, c=chorsen_df["corpus_id"], cmap="Blues")
    ax2.set_xlabel("RC")
    ax2.set_ylabel('NCTC')
    # ax2.set_title("r-nct")

    ax3 = plt.subplot(212)
    ax3 = sns.scatterplot(x=chorsen_df["mean_ct_chromaticity"].tolist(),
                          y=chorsen_df["mean_nct_chromaticity"].tolist(),
                          s=5, c=chorsen_df["corpus_id"], cmap="Oranges")
    ax3.set_xlabel('CTC')
    ax3.set_ylabel('NCTC')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_chromaticities.tsv", sep="\t")
    plot_chromaticity_correlation(result_df, period="renaissance")
