from typing import Tuple, Literal

import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt

from analysis import get_period_df, correlation, get_bwv808_example_CI, get_corpuswise_fifth_range


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


# CI correlation _______________________________________________________
def plot_CI_corr_byPeriod(df: pd.DataFrame, corr_method: Literal["pearson", "spearman"],
                          fig_name: Literal["MajorMode", "MinorMode", "CombinedMode"]):
    t1, t2, t3, t4 = (1662, 1763, 1821, 1869)

    fig, axs = plt.subplots(3, 5, layout="constrained", figsize=(18, 10))

    colors = ["#580F41", "#173573", "#c4840c"]
    periods = ["renaissance", "baroque", "classical", "early_romantic", "late_romantic"]
    years = [f"<{t1}", f"{t1}-{t2}", f"{t2}-{t3}", f"{t3}-{t4}", f">{t4}"]

    for i, period in enumerate(periods):
        period_df = get_period_df(df, period)

        # Set column subtitle
        axs[0, i].set_title(years[i], fontweight="bold", fontsize=18, family="sans-serif")

        for j, (x_var, y_var) in enumerate([("RC", "CTC"),
                                            ("RC", "NCTC"),
                                            ("CTC", "NCTC")]):
            if corr_method == "pearson":
                g = sns.regplot(ax=axs[j, i], data=period_df, x=x_var, y=y_var, color=colors[j])
            elif corr_method == "spearman":
                g = sns.regplot(ax=axs[j, i], data=period_df, x=x_var, y=y_var, color=colors[j],
                                # fit_reg=False,
                                lowess=True)
            else:
                raise ValueError
            g.set_xlabel(f"{x_var}", fontsize=15)
            g.set_ylabel(f"{y_var}", fontsize=15)

            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            # r = round(correlation(period_df[x_var], period_df[y_var])["r"].values[0], 2)
            r = pg.corr(period_df[x_var], period_df[y_var], method=corr_method).round(3)["r"].values[0]
            p = pg.corr(period_df[x_var], period_df[y_var], method=corr_method).round(3)["p-val"].values[0]

            if p < 0.001:
                p_text = 'p < .001'

            elif 0.001 < p < 0.05:
                p_text = 'p < .05'

            else:
                p_text = f'p = {p:.2f}'

            # adding the text
            x_limit = axs[j, i].get_xlim()
            y_limit = axs[j, i].get_ylim()
            x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
            y_pos_1 = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

            if corr_method == "pearson":
                g.text(x_pos, y_pos_1, f'r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right',
                       va='top')
            else:
                g.text(x_pos, y_pos_1, f'{corr_method} r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right',
                   va='top')

    plt.savefig(f"figs/Figure_CI_corr_byPeriod_{corr_method}_{fig_name}.pdf", dpi=300)

    plt.show()


def plot_CI_corr_byMode(df: pd.DataFrame, mode: Literal["MajorMode", "MinorMode"]):
    fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))

    # axs[0, i].set_title(years[i], fontweight="bold", fontsize=18, family="sans-serif")
    colors = ["#580F41", "#173573", "#c4840c"]

    for j, (x_var, y_var) in enumerate([("RC", "CTC"),
                                        ("RC", "NCTC"),
                                        ("CTC", "NCTC")]):
        g = sns.regplot(ax=axs[j], data=df, x=x_var, y=y_var, color=colors[j])
        g.set_xlabel(f"{x_var}", fontsize=15)
        g.set_ylabel(f"{y_var}", fontsize=15)

        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        r = round(correlation(df[x_var], df[y_var])["r"].values[0], 2)
        p = correlation(df[x_var], df[y_var])["p-val"].values[0]

        if p < 0.001:
            p_text = 'p < .001'
        elif 0.001 < p < 0.05:
            p_text = 'p < .05'
        else:
            p_text = f'p = {p:.2f}'

        # adding the text
        x_limit = axs[j].get_xlim()
        y_limit = axs[j].get_ylim()
        x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
        y_pos = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

        g.text(x_pos, y_pos, f'r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right', va='top')

    plt.savefig(f"figs/Figure_CI_corr_byMode_{mode}.pdf", dpi=300)

    plt.show()


# MUSICAL EXAMPLES ____________________________________________________
def plot_bwv808_CI_comparison():
    original = get_bwv808_example_CI(version="original")
    agrements = get_bwv808_example_CI(version="agrements")

    df = pd.concat([original, agrements])

    long_df = df.reset_index().melt(id_vars='index', var_name='Type', value_name='Value')
    long_df = long_df.rename(columns={'index': 'piece', 'Type': 'CI Type', 'Value': 'CI Value'})

    print(long_df)

    ax = sns.barplot(long_df, x="CI Type", y="CI Value", hue="piece", palette=["#194F67", "#CFB980"])
    ax.set_xlabel("", fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=13)

    ax.legend(title=None, fontsize=13)

    plt.tight_layout()
    plt.savefig("figs/Figure_BWV808_CI_results.pdf", format='pdf')




# FIFTHS RANGE PLOT ____________________________________________________
def plot_corpus_fifths_range(piece:pd.DataFrame):

    corpus = get_corpuswise_fifth_range(piece)
    catplot=sns.catplot(
        data=corpus, x="corpus", y="FR_value", hue="FR_type",
        palette={"max_r_5thRange": "g", "max_ct_5thRange": "r", "max_nct_5thRange": "m"},
        markers=["o", "o", "o"], linestyles=["-", "-", "-"],
        kind="point",
        height=6,  # Adjust the height if needed
        aspect=2  # Adjust the aspect ratio to make the figure wider
    )
    catplot.set_xticklabels(rotation=45)

    plt.show()


if __name__ == "__main__":
    combined = pd.read_csv("data/piece_indices.tsv", sep="\t")

    # piece_distribution_with_EraDivision(result_df, show_fig=True, save_histogram=False)

    major = pd.read_csv("data/majorkey_piece_indices.tsv", sep="\t")
    minor = pd.read_csv("data/minorkey_piece_indices.tsv", sep="\t")
    #
    # plot_CI_corr_byPeriod(df=major, fig_name="MajorMode")
    # plot_CI_corr_byPeriod(df=minor, fig_name="MinorMode")
    # plot_CI_corr_byPeriod(df=combined, fig_name="CombinedMode")

    # plot_CI_corr_byPeriod(df=major, fig_name="MajorMode", corr_method="pearson")
    # plot_CI_corr_byPeriod(df=minor, fig_name="MinorMode", corr_method="pearson")

    plot_corpus_fifths_range(combined)
