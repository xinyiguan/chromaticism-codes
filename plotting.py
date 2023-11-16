import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from analysis import get_period_df, correlation, get_bwv808_example_CI, get_k331_1_example_CI


def piece_distribution_histogram(df: pd.DataFrame, save_fig: bool = False):
    # global variables
    DPI = 300

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

    t1 = round(t1)
    t2 = round(t2)
    t3 = round(t3)
    t4 = round(t4)

    print(f'{t1=}, {t2=}, {t3=}, {t4=} ')

    # Set labels and show the plot
    plt.xlabel("year")
    plt.ylabel("probability")
    if save_fig:
        plt.savefig("figs/Figure_histogram.pdf", dpi=DPI)
    plt.show()
    return t1, t2, t3, t4




def plot_chromaticity_indices_corr(df: pd.DataFrame):
    # # rename the col names of CI in the df:
    # df = df.rename(columns={'rc': 'RC',
    #                         'ctc': 'CTC',
    #                         'nctc': 'NCTC'})

    fig, axs = plt.subplots(3, 5, layout="constrained", figsize=(18, 10))

    colors = ["#580F41", "#173573", "#c4840c"]
    periods = ["renaissance", "baroque", "classical", "early_romantic", "late_romantic"]
    years = ["<1662", "1662-1761", "1761-1820", "1820-1869", ">1869"]

    for i, period in enumerate(periods):
        period_df = get_period_df(df, period)

        # Set column subtitle
        axs[0, i].set_title(years[i], fontweight="bold", size="x-large", family="sans-serif")

        for j, (x_var, y_var) in enumerate([("RC", "CTC"),
                                            ("RC", "NCTC"),
                                            ("CTC", "NCTC")]):
            g = sns.regplot(ax=axs[j, i], data=period_df, x=x_var, y=y_var, color=colors[j])

            r = round(correlation(period_df[x_var], period_df[y_var])["r"].values[0], 3)
            p = correlation(period_df[x_var], period_df[y_var])["p-val"].values[0]

            if p < 0.001:
                p_text = 'p < .001'
            elif 0.001 < p < 0.05:
                p_text = 'p < .05'
            else:
                p_text = f'p = {p:.3f}'

            # adding the text
            x_limit = axs[j, i].get_xlim()
            y_limit = axs[j, i].get_ylim()
            x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
            y_pos = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

            g.text(x_pos, y_pos, f'r = {r}, {p_text}', fontstyle='italic', ha='right', va='top')

    plt.savefig("figs/Figure_CI_corr.pdf", dpi=300)

    plt.show()


def plot_bwv808_CI_comparison():

    original = get_bwv808_example_CI(version="original")
    agrements = get_bwv808_example_CI(version="agrements")

    df = pd.concat([original, agrements])

    long_df = df.reset_index().melt(id_vars='index', var_name='Type', value_name='Value')
    long_df = long_df.rename(columns={'index': 'piece', 'Value': 'CI Value'})


    print(long_df)

    sns.barplot(long_df, x="Type", y="CI Value", hue="piece")
    plt.tight_layout()
    plt.savefig("figs/Figure_BWV808_CI_results.pdf", format='pdf')


def plot_k331_CI_comparison():



    thema = get_k331_1_example_CI(version="thema")
    var5 = get_k331_1_example_CI(version="var5")
    var6 = get_k331_1_example_CI(version="var6")

    df = pd.concat([thema, var5, var6])
    # with pd.option_context('display.max_columns', None):
    #     print(df)

    long_df = df.reset_index().melt(id_vars='index', var_name='Type', value_name='Value')
    long_df = long_df.rename(columns={'index': 'version', 'Value': 'CI Value'})


    # long_df = pd.melt(df, id_vars=['chord', "version"],
    #          value_vars=['r_chromaticity', 'ct_chromaticity', 'nct_chromaticity'],
    #          var_name='CI Type', value_name='CI Value')

    # long_df_content = pd.melt(df, id_vars=['chord', "version"],
    #          value_vars=['root', 'ct', 'nct'],
    #          var_name='Content Type', value_name='Content Value')
    #
    # long_df = pd.merge(long_df_CI, long_df_content, on=['chord', 'version'])
    # long_df = long_df[["version", "chord", "Content Type", "Content Value", "CI Type", "CI Value"]]

    # with pd.option_context('display.max_columns', None):
    #     print(long_df)
    #     long_df.to_csv("k331_CI_long.tsv", sep="\t")


    # sns.barplot(df, x="CI Type", y="CI Value", hue="version")
    sns.barplot(long_df, x="Type", y="CI Value", hue="version")

    plt.tight_layout()
    plt.show()
    # plt.savefig("figs/Figure_K331_CI_results.pdf", format='pdf')



if __name__ == "__main__":
    # result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    # piece_distribution_histogram(result_df, save_fig=False)

    plot_k331_CI_comparison()
