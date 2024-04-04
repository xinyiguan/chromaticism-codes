import os
from typing import Literal

import pandas as pd

from Code.utils.util import load_file_as_df

# %% Analysis: piece distribution in periods
import os
from typing import Literal
import pandas as pd
import seaborn as sns
import pingouin as pg

from matplotlib import pyplot as plt

from Code.utils.util import load_file_as_df, corpus_composer_dict, corpus_collection_dict
from Code.utils.auxiliary import create_results_folder, determine_period_Johannes, determine_period, \
    determine_period_id, get_period_df_Johannes, get_period_df


# %% Analysis: Piece distributions fig and table
def piece_distribution(df: pd.DataFrame, period_by: Literal["Johannes", "old"], repo_dir: str):
    """
    df: dissonance_piece_average (corpus, piece, period, period_Johannes)
    """
    # save the results to this folder:
    result_dir = create_results_folder(analysis_name="piece_distribution", repo_dir=repo_dir)

    pieces_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        # period=("period", "first"),
        # period_Johannes=("period_Johannes", "first"),
    )

    pieces_df["period_Johannes"] = pieces_df.apply(determine_period_Johannes, axis=1)
    pieces_df["period"] = pieces_df.apply(determine_period, axis=1)

    pieces_df["corpus_id"] = pd.factorize(pieces_df["corpus"])[0] + 1
    pieces_df["piece_id"] = pd.factorize(pieces_df["piece"])[0] + 1
    pieces_df["period_id"] = pieces_df.apply(determine_period_id, axis=1)

    metainfo_df = pd.DataFrame.from_dict(corpus_composer_dict, orient='index').reset_index()
    metainfo_df["Collection"] = metainfo_df['index'].map(corpus_collection_dict)
    metainfo_df = metainfo_df.rename(columns={'index': 'corpus', 0: 'Composer'})

    if period_by == "Johannes":

        ## plotting ________________________
        h = sns.histplot(pieces_df["piece_year"], kde=True, stat="probability", bins=40,
                         kde_kws={'bw_adjust': 0.6})
        h.set_xlabel("Year", fontsize=15)
        h.set_ylabel("probability", fontsize=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        for b in [1650, 1750, 1800]:
            h.axvline(b, c="gray", ls="--", zorder=-2)

        # save the fig
        p = h.get_figure()
        p.savefig(f"{result_dir}fig_histogram_Johannes_period.pdf", dpi=300)

        ## stats __________________________
        piece_num = pieces_df.groupby(['corpus', 'period_Johannes']).agg(
            Piece_Number=('piece', 'count'),
            corpus_id=('corpus_id', "first"),
            period_id=('period_id', 'first')
        ).reset_index().sort_values(by=["corpus_id", "period_id"])

        stats_df = pd.merge(piece_num, metainfo_df, on='corpus', how='left')
        stats_df = stats_df[["period_Johannes", "Composer", "corpus", "Piece_Number"]]
        stats_df = stats_df.rename(
            columns={"period_Johannes": "Period", "corpus": "Corpus", "Piece_Number": "Piece Number"})
        stats_df['Corpus'] = stats_df['Corpus'].str.replace('_', ' ')
        stats_df.to_pickle(f'{result_dir}corpus_stats_table_JP.pickle')
        stats_df.to_latex(buf=f'{result_dir}corpus_stats_latex_table_JP')

    elif period_by == "old":

        ## plotting ________________________
        h = sns.histplot(pieces_df["piece_year"], kde=True, stat="probability", bins=40,
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

        # save time-period division
        period_division_txt = [f't1={round(t1)}', f't2={round(t2)}', f't3={round(t3)}', f't4={round(t4)}']
        with open(f'{result_dir}old_period_division.txt', 'w') as f:
            f.write('\n'.join(period_division_txt))

        # save the fig
        p = h.get_figure()
        p.savefig(f"{result_dir}fig_histogram_period.pdf", dpi=300)

        ## stats __________________________
        piece_num = pieces_df.groupby(['corpus', 'period']).agg(
            Piece_Number=('piece', 'count'),
            corpus_id=('corpus_id', "first"),
            period_id=('period_id', 'first')
        ).reset_index().sort_values(by=["corpus_id", "period_id"])

        stats_df = pd.merge(piece_num, metainfo_df, on='corpus', how='left')
        stats_df = stats_df[["period", "Composer", "corpus", "Piece_Number"]]
        stats_df = stats_df.rename(
            columns={"period_Johannes": "Period", "corpus": "Corpus", "Piece_Number": "Piece Number"})
        stats_df['Corpus'] = stats_df['Corpus'].str.replace('_', ' ')
        stats_df.to_pickle(f'{result_dir}corpus_stats_table.pickle')
        stats_df.to_latex(buf=f'{result_dir}corpus_stats_table')

    else:
        raise ValueError


# # main _________________________________________________________________________
# repo_dir = '/Users/xguan/Codes/chromaticism-codes/'
# df = load_file_as_df(path=f"{repo_dir}/Data/prep_data/processed_DLC_data.pickle")
# piece_distribution(df=df, period_by="old", repo_dir=repo_dir)
# print(f'Finished piece distribution analysis for old period division...')
# piece_distribution(df=df, period_by="Johannes", repo_dir=repo_dir)
# print(f'Finished piece distribution analysis for new (Johannes) period division...')


# %% Analysis: Chromaticity-Dissonance: correlation between chord-level WLC and WLD

def corr_chord_level_WLC_WLD(df: pd.DataFrame,
                             period_by: Literal["Johannes", "old"]):
    """
    df: the chord-level indices dataframe containing all chord-level WLC and WLD values
    """
    # save the results to this folder:
    result_dir = create_results_folder(analysis_name="chromaticity_dissonance_corr", repo_dir=repo_dir)

    # A1: corr by Johannes period _____________________________________

    if period_by == "Johannes":
        ## static fig:
        fig, axs = plt.subplots(1, 4, layout="constrained", figsize=(18, 5))
        colors = ["#580F41", "#173573", "#c4840c", "#176373"]
        J_periods = ["pre-Baroque", "Baroque", "Classical", "Extended tonality"]
        J_years = [f"<1650", f"1650-1750", f"1750-1800", f">1800"]

        for i, period in enumerate(J_periods):
            period_df = get_period_df_Johannes(df, period)

            # Set column subtitle
            axs[i].set_title(J_years[i], fontweight="bold", fontsize=18, family="sans-serif")
            g = sns.regplot(ax=axs[i], data=period_df, x="WLC", y="WLD", color=colors[i])
            g.set_xlabel(f"WLC", fontsize=15)
            g.set_ylabel(f"WLD", fontsize=15)

            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            r = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["r"].values[0]
            p = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["p-val"].values[0]

            if p < 0.001:
                p_text = 'p < .001'
            elif 0.001 < p < 0.05:
                p_text = 'p < .05'
            else:
                p_text = f'p = {p:.2f}'

            # adding the text
            x_limit = axs[i].get_xlim()
            y_limit = axs[i].get_ylim()
            x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
            y_pos_1 = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

            g.text(x_pos, y_pos_1, f'r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right',
                   va='top')

        fig.savefig(f"{result_dir}fig_WLC_WLD_corr_by_JPeriods.pdf", dpi=200)


    # A2: corr by old period _____________________________________
    else:
        ## static fig:
        fig, axs = plt.subplots(1, 5, layout="constrained", figsize=(22, 5))
        colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
        t1, t2, t3, t4 = (1662, 1763, 1821, 1869)
        years = [f"<{t1}", f"{t1}-{t2}", f"{t2}-{t3}", f"{t3}-{t4}", f">{t4}"]
        periods = ["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]

        for i, period in enumerate(periods):
            period_df = get_period_df(df, period)

            # Set column subtitle
            axs[i].set_title(years[i], fontweight="bold", fontsize=18, family="sans-serif")
            g = sns.regplot(ax=axs[i], data=period_df, x="WLC", y="WLD", color=colors[i])
            g.set_xlabel(f"WLC", fontsize=15)
            g.set_ylabel(f"WLD", fontsize=15)

            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            r = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["r"].values[0]
            p = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["p-val"].values[0]

            if p < 0.001:
                p_text = 'p < .001'
            elif 0.001 < p < 0.05:
                p_text = 'p < .05'
            else:
                p_text = f'p = {p:.2f}'

            # adding the text
            x_limit = axs[i].get_xlim()
            y_limit = axs[i].get_ylim()
            x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
            y_pos_1 = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

            g.text(x_pos, y_pos_1, f'r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right',
                   va='top')

        fig.savefig(f"{result_dir}fig_WLC_WLD_corr_by_Periods.pdf", dpi=200)


# main _________________________________________________________________________
# repo_dir = '/Users/xguan/Codes/chromaticism-codes/'
# df = load_file_as_df(path=f"{repo_dir}/Data/prep_data/for_analysis/chord_level_indices.pickle")
# corr_chord_level_WLC_WLD(df=df, period_by="Johannes")
# corr_chord_level_WLC_WLD(df=df, period_by="old")


# %% Analysis: Chromaticity-correlation between WLC and OLC

def piece_chromaticity_WLC_OLC_corr(df: pd.DataFrame, period_by: Literal["Johannes", "old"]):
    """
    df: take chromaticity_piece_by_mode df
    """
    # save the results to this folder:
    result_dir = create_results_folder(analysis_name="piece_chromatcities_corr", repo_dir=repo_dir)

    # stats: two-sample t-test for major/minor mode segments

    major_df = df[df['localkey_mode'].isin(['major'])]
    minor_df = df[df['localkey_mode'].isin(['minor'])]

    WLC_major = major_df.loc[:, "WLC"]
    WLC_minor = minor_df.loc[:, "WLC"]

    OLC_major = major_df.loc[:, "OLC"]
    OLC_minor = minor_df.loc[:, "OLC"]

    WLC_ttest_result = pg.ttest(x=WLC_major, y=WLC_minor).rename(index={'T-test': 'WLC'})
    OLC_ttest_result = pg.ttest(x=OLC_major, y=OLC_minor).rename(index={'T-test': 'OLC'})

    ttest_result = pd.concat([WLC_ttest_result, OLC_ttest_result])
    ttest_result.to_latex(f'{result_dir}mode_ttest_result.txt')

    # WLC and OLC values across time
    pd.DataFrame.from_dict(data={
        'WLC(major)': WLC_major.mean(),
        'WLC(minor)': WLC_minor.mean(),
        'OLC(major)': OLC_major.mean(),
        'OLC(minor)': OLC_minor.mean()},
        orient='index'
    ).reset_index().rename(columns={"index": "CI Type", 0: "Value"}).to_latex(f'{result_dir}agg_CI_vals_by_mode.txt',
                                                                              index=False)

    # plot by mode segment and by period.

    if period_by == "Johannes":
        ## static fig:
        fig, axs = plt.subplots(2, 4, layout="constrained", figsize=(18, 8))
        colors = ["#580F41", "#173573", "#c4840c", "#176373"]
        J_periods = ["pre-Baroque", "Baroque", "Classical", "Extended tonality"]
        J_years = [f"<1650", f"1650-1750", f"1750-1800", f">1800"]

        for i, period in enumerate(J_periods):
            major_period_df = get_period_df_Johannes(major_df, period)
            minor_period_df = get_period_df_Johannes(minor_df, period)

            # Set column subtitle
            axs[0, i].set_title(J_years[i], fontweight="bold", fontsize=18, family="sans-serif")
            ma = sns.regplot(ax=axs[0, i], data=major_period_df, x="WLC", y="OLC", color=colors[i])
            mi = sns.regplot(ax=axs[1, i], data=minor_period_df, x="WLC", y="OLC", color=colors[i])

            plt.xlabel(f"WLC", fontsize=15)
            plt.ylabel(f"OLC", fontsize=15)

            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            major_r = pg.corr(major_period_df["WLC"], major_period_df["OLC"], method="pearson").round(3)["r"].values[0]
            major_p = pg.corr(major_period_df["WLC"], major_period_df["OLC"], method="pearson").round(3)["p-val"].values[0]
            if major_p < 0.001:
                major_p_txt = 'p < .001'
            elif 0.001 < major_p < 0.05:
                major_p_txt = 'p < .05'
            else:
                major_p_txt = f'p = {major_p:.2f}'

            minor_r = pg.corr(minor_period_df["WLC"], minor_period_df["OLC"], method="pearson").round(3)["r"].values[0]
            minor_p = pg.corr(minor_period_df["WLC"], minor_period_df["OLC"], method="pearson").round(3)["p-val"].values[0]
            if minor_p < 0.001:
                minor_p_txt = 'p < .001'
            elif 0.001 < minor_p < 0.05:
                minor_p_txt = 'p < .05'
            else:
                minor_p_txt = f'p = {minor_p:.2f}'


            # adding the text
            ma_x_limit = axs[0, i].get_xlim()
            mi_x_limit = axs[1, i].get_xlim()

            ma_y_limit = axs[0, i].get_ylim()
            mi_y_limit = axs[1, i].get_ylim()

            ma_x_pos = ma_x_limit[1] - 0.03 * (ma_x_limit[1] - ma_x_limit[0])
            mi_x_pos = mi_x_limit[1] - 0.03 * (mi_x_limit[1] - mi_x_limit[0])

            ma_y_pos = ma_y_limit[1] - 0.03 * (ma_y_limit[1] - ma_y_limit[0])
            mi_y_pos = mi_y_limit[1] - 0.03 * (mi_y_limit[1] - mi_y_limit[0])


            ma.text(ma_x_pos, ma_y_pos, f'r = {major_r}, {major_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                   va='top')
            mi.text(mi_x_pos, mi_y_pos, f'r = {minor_r}, {minor_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                   va='top')

        axs[0, 0].set_ylabel(f"Major Mode", fontsize=15)
        axs[1, 0].set_ylabel(f"Minor Mode", fontsize=15)
        fig.supxlabel("WLC")
        fig.supylabel("OLC")
        fig.savefig(f"{result_dir}fig_WLC_OLC_corr_by_JPeriod.pdf", dpi=200)


    else:
        fig, axs = plt.subplots(2, 5, layout="constrained", figsize=(22, 8))
        colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
        periods = ["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]
        years = [f"<1662", f"1662-1763", f"1763-1821", f"1821-1869", f">1869"]

        for i, period in enumerate(periods):
            major_period_df = get_period_df(major_df, period)
            minor_period_df = get_period_df(minor_df, period)

            # Set column subtitle
            axs[0, i].set_title(years[i], fontweight="bold", fontsize=18, family="sans-serif")
            ma = sns.regplot(ax=axs[0, i], data=major_period_df, x="WLC", y="OLC", color=colors[i])
            mi = sns.regplot(ax=axs[1, i], data=minor_period_df, x="WLC", y="OLC", color=colors[i])

            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            major_r = pg.corr(major_period_df["WLC"], major_period_df["OLC"], method="pearson").round(3)["r"].values[0]
            major_p = \
            pg.corr(major_period_df["WLC"], major_period_df["OLC"], method="pearson").round(3)["p-val"].values[0]
            if major_p < 0.001:
                major_p_txt = 'p < .001'
            elif 0.001 < major_p < 0.05:
                major_p_txt = 'p < .05'
            else:
                major_p_txt = f'p = {major_p:.2f}'

            minor_r = pg.corr(minor_period_df["WLC"], minor_period_df["OLC"], method="pearson").round(3)["r"].values[0]
            minor_p = \
            pg.corr(minor_period_df["WLC"], minor_period_df["OLC"], method="pearson").round(3)["p-val"].values[0]
            if minor_p < 0.001:
                minor_p_txt = 'p < .001'
            elif 0.001 < minor_p < 0.05:
                minor_p_txt = 'p < .05'
            else:
                minor_p_txt = f'p = {minor_p:.2f}'

            # adding the text
            ma_x_limit = axs[0, i].get_xlim()
            mi_x_limit = axs[1, i].get_xlim()

            ma_y_limit = axs[0, i].get_ylim()
            mi_y_limit = axs[1, i].get_ylim()

            ma_x_pos = ma_x_limit[1] - 0.03 * (ma_x_limit[1] - ma_x_limit[0])
            mi_x_pos = mi_x_limit[1] - 0.03 * (mi_x_limit[1] - mi_x_limit[0])

            ma_y_pos = ma_y_limit[1] - 0.03 * (ma_y_limit[1] - ma_y_limit[0])
            mi_y_pos = mi_y_limit[1] - 0.03 * (mi_y_limit[1] - mi_y_limit[0])

            ma.text(ma_x_pos, ma_y_pos, f'r = {major_r}, {major_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                    va='top')
            mi.text(mi_x_pos, mi_y_pos, f'r = {minor_r}, {minor_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                    va='top')

        axs[0, 0].set_ylabel(f"Major Mode", fontsize=15)
        axs[1, 0].set_ylabel(f"Minor Mode", fontsize=15)
        fig.supxlabel("WLC")
        fig.supylabel("OLC")

        fig.savefig(f"{result_dir}fig_WLC_OLC_corr_by_period.pdf", dpi=200)




if __name__ == "__main__":
    repo_dir = '/Users/xguan/Codes/chromaticism-codes/'
    df = load_file_as_df(path=f"{repo_dir}/Data/prep_data/for_analysis/chromaticity_piece_by_mode.pickle")
    piece_chromaticity_WLC_OLC_corr(df=df, period_by="Johannes")
