import os
from fractions import Fraction
from typing import Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

from Code.utils.util import load_file_as_df, corpus_composer_dict, corpus_collection_dict, corpus_prettyprint_dict, \
    save_df
from Code.utils.auxiliary import create_results_folder, determine_period_id, get_period_df, Fabian_periods, \
    pprint_p_text, get_piece_df_by_localkey_mode, exclude_piece_from_corpus
import plotly.express as px


# %% Analysis: Basic chrom, diss stats

def stats_by_piece(df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    """
    df: the chord level indices
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="basic_stats", repo_dir=repo_dir)

    pieces_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        period=("period", "first"),

        min_ILC=("ILC", 'min'),
        avg_ILC=("ILC", 'mean'),
        max_ILC=("ILC", 'max'),

        min_OLC=("OLC", 'min'),
        avg_OLC=("OLC", 'mean'),
        max_OLC=("OLC", 'max'),

        min_ILD=("ILD", 'min'),
        avg_ILD=("ILD", 'mean'),
        max_ILD=("ILD", 'max')
    )

    pieces_df.to_latex(f'{result_dir}piece_level_indices_stats.txt')

    return pieces_df


def stats_by_piece_by_mode(piece_level_indices_df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="basic_stats",
                                       repo_dir=repo_dir)

    major_df = get_piece_df_by_localkey_mode(df=piece_level_indices_df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=piece_level_indices_df, mode="minor")

    ILC_major = major_df["ILC"].mean()
    ILC_minor = minor_df["ILC"].mean()
    OLC_major = major_df["OLC"].mean()
    OLC_minor = minor_df["OLC"].mean()
    ILD_major = major_df["ILD"].mean()
    ILD_minor = minor_df["ILD"].mean()

    data = {
        "Index": ["ILC", "ILC", "OLC", "OLC", "ILD", "ILD"],
        "mode": ["major", "minor", "major", "minor", "major", "minor"],
        "value": [ILC_major, ILC_minor, OLC_major, OLC_minor, ILD_major, ILD_minor]
    }

    df = pd.DataFrame(data=data)
    df = pd.pivot(df, index='Index', columns='mode', values='value')
    df.to_latex(f'{result_dir}piece_level_indices_by_mode_stats.txt')
    return df


def stats_by_corpus(df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    """
    df: the chord level indices
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="basic_stats", repo_dir=repo_dir)

    corpora_df = df.groupby(['corpus'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),

        min_ILC=("ILC", 'min'),
        avg_ILC=("ILC", 'mean'),
        max_ILC=("ILC", 'max'),

        min_OLC=("OLC", 'min'),
        avg_OLC=("OLC", 'mean'),
        max_OLC=("OLC", 'max'),

        min_ILD=("ILD", 'min'),
        avg_ILD=("ILD", 'mean'),
        max_ILD=("ILD", 'max')
    )

    corpora_df.to_latex(f'{result_dir}corpora_level_indices_stats.txt')

    return corpora_df


def DLC_corpus_stats(chord_level_df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    """
    This function computes the number of:
    pieces, chord symbols, unique chord tokens.
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="basic_stats", repo_dir=repo_dir)

    df = chord_level_df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"))

    df = df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)

    df["corpus_id"] = pd.factorize(df["corpus"])[0] + 1
    df["piece_id"] = pd.factorize(df["piece"])[0] + 1

    num_subcorpora = df["corpus_id"].max()
    num_pieces = df["piece_id"].max()
    num_chords = chord_level_df.shape[0]
    num_unique_chords = chord_level_df["chord"].unique().shape[0]

    df_by_mode_seg = chord_level_df.groupby(["corpus", "piece", "localkey_mode"], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        localkey_mode=("localkey_mode", "first")
    )
    num_major_seg = df_by_mode_seg.localkey_mode.value_counts()['major']
    num_minor_seg = df_by_mode_seg.localkey_mode.value_counts()['minor']

    percentage_major_seg = num_major_seg / (num_major_seg + num_minor_seg)
    percentage_minor_seg = num_minor_seg / (num_major_seg + num_minor_seg)

    result = pd.Series({
        "num sub-corpora": num_subcorpora,
        "num pieces": num_pieces,
        "num chords": num_chords,
        "num unique chord": num_unique_chords,
        "num major segments": num_major_seg,
        "num minor segments": num_minor_seg,
        "percentage major segments": percentage_major_seg,
        "percentage minor segments": percentage_minor_seg
    }).to_frame()

    result.to_string(f'{result_dir}DLC_corpus_stats.txt')

    return result


def plot_chord_size_across_time(df: pd.DataFrame):
    """
    df: assuming we take the preprocessed_DLC df
    """

    df['chord_size'] = df.chord_tones.apply(lambda x: len(x))
    sns.jointplot(data=df, x="piece_year", y="chord_size", )
    plt.show()


# %% Analysis: Piece distributions fig and table
def piece_distribution(df: pd.DataFrame, repo_dir: str) -> None:
    """
    df: processed_DLC_data (corpus, piece, period, period_Johannes)
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="piece_distribution", repo_dir=repo_dir)

    pieces_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        period=("period", "first")
    )

    pieces_df["corpus_id"] = pd.factorize(pieces_df["corpus"])[0] + 1
    pieces_df["piece_id"] = pd.factorize(pieces_df["piece"])[0] + 1
    pieces_df["period_id"] = pieces_df.apply(lambda row: determine_period_id(row=row, method="Fabian"), axis=1)

    metainfo_df = pd.DataFrame.from_dict(corpus_composer_dict, orient='index').reset_index()
    metainfo_df["Collection"] = metainfo_df['index'].map(corpus_collection_dict)
    metainfo_df = metainfo_df.rename(columns={'index': 'corpus', 0: 'Composer'})

    # general plotting setup
    h = sns.histplot(pieces_df["piece_year"], kde=True, stat="probability", bins=40,
                     kde_kws={'bw_adjust': 0.6})
    h.set_xlabel("Year", fontsize=15)
    h.set_ylabel("probability", fontsize=15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ## plotting ________________________
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
    for v in [t1, t2, t3, t4]:
        h.axvline(v, c="gray", ls="--")

    # save time-period division
    period_division_txt = [f't1={round(t1)}', f't2={round(t2)}', f't3={round(t3)}', f't4={round(t4)}']
    with open(f'{result_dir}FabianPeriod_division.txt', 'w') as f:
        f.write('\n'.join(period_division_txt))

    plt.tight_layout()
    plt.savefig(f"{result_dir}fig_histogram_FP.pdf", dpi=200)

    ## stats __________________________
    piece_num = pieces_df.groupby(['corpus', 'period']).agg(
        Piece_Number=('piece', 'count'),
        corpus_id=('corpus_id', "first"),
        period_id=('period_id', 'first')
    ).reset_index().sort_values(by=["corpus_id", "period_id"])

    stats_df = pd.merge(piece_num, metainfo_df, on='corpus', how='left')
    stats_df = stats_df[["period", "Composer", "corpus", "Piece_Number"]]
    stats_df = stats_df.rename(
        columns={"period": "Period", "corpus": "Corpus", "Piece_Number": "Piece Number"})
    stats_df['Corpus'] = stats_df['Corpus'].str.replace('_', ' ')
    stats_df.to_pickle(f'{result_dir}corpus_stats_table_FP.pickle')
    stats_df.to_latex(buf=f'{result_dir}corpus_stats_table_FP.txt')
    del h


# %% Analysis: Correlation analyses - Piece-level (global indices)

def _global_indices_pair_correlation_stats(df: pd.DataFrame,
                                           indices_pair: Tuple[
                                               Literal["ILC"], Literal["OLC", "ILD"]],
                                           method: Literal["pearson", "spearman"]) -> Tuple[float, float]:
    """
    df: assuming we take the piece-level df
    """
    if indices_pair == ("ILC", "OLC") or ("ILC", "ILD"):
        idx1, idx2 = indices_pair
        r = pg.corr(df[idx1], df[idx2], method=method).round(3)["r"].values[0]
        p_val = pg.corr(df[idx1], df[idx2], method=method).round(3)["p-val"].values[0]
    else:
        raise NotImplementedError
    return r, p_val


def _global_indices_corr_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
                                 indices_pair: Tuple[
                                     Literal["ILC"], Literal["OLC", "ILD"]],
                                 corr_method: Literal["pearson", "spearman"],
                                 out_type: Literal["df", "tuple"] = "df") -> pd.DataFrame | Tuple:
    major_r, major_p = _global_indices_pair_correlation_stats(df=major_df, indices_pair=indices_pair,
                                                              method=corr_method)
    minor_r, minor_p = _global_indices_pair_correlation_stats(df=minor_df, indices_pair=indices_pair,
                                                              method=corr_method)

    if out_type == "df":
        stats = pd.DataFrame(data=[[major_r, major_p, minor_r, minor_p]], columns=["r (major segment)",
                                                                                   "p (major segment)",
                                                                                   "r (minor segment)",
                                                                                   "p (minor segment)"])
    elif out_type == "tuple":
        stats = (major_r, major_p, minor_r, minor_p)
    else:
        raise NotImplementedError
    return stats


def compute_piece_corr_stats(df: pd.DataFrame,
                             indices_pair: Tuple[
                                 Literal["ILC"], Literal["OLC", "ILD"]],
                             repo_dir: str,
                             corr_method: Literal["pearson", "spearman"],
                             outliers_to_exclude_major: Optional[List[Tuple[str, str]]],
                             outliers_to_exclude_minor: Optional[List[Tuple[str, str]]],
                             save: bool) -> pd.DataFrame:
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="correlation_analyses",
                                       repo_dir=repo_dir)

    major_df = get_piece_df_by_localkey_mode(df=df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor")

    # # manually remove outliers -------:
    # ## current criteria: OLC >10 OR ILC >10
    # major_ER = [("chopin_mazurkas", "BI162-3op63-3"), ("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca")]
    # major_LR = [("tchaikovsky_seasons", "op37a12"), ("dvorak_silhouettes", "op08n12"),
    #             ("dvorak_silhouettes", "op08n01")]
    #
    # minor_ER = [("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca")]
    # minor_LR = [("bartok_bagatelles", "op06n12")] # ILC>10
    # # end of outlier list -------------

    if outliers_to_exclude_major:
        major_df = exclude_piece_from_corpus(df=major_df, corpus_piece_tups=outliers_to_exclude_major)
    else:
        major_df = major_df
    if outliers_to_exclude_minor:
        minor_df = exclude_piece_from_corpus(df=minor_df, corpus_piece_tups=outliers_to_exclude_minor)
    else:
        minor_df = minor_df

    result_dfs = []
    for p in Fabian_periods:
        major_df_p = get_period_df(df=major_df, method="Fabian", period=p)
        minor_df_p = get_period_df(df=minor_df, method="Fabian", period=p)

        stats = _global_indices_corr_by_mode(major_df=major_df_p, minor_df=minor_df_p, indices_pair=indices_pair,
                                             corr_method=corr_method)
        stats["period"] = p
        result_dfs.append(stats)
    result_df = pd.concat(result_dfs)

    if save:
        idx1, idx2 = indices_pair
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        result_df.to_latex(buf=f'{sub_corr_analysis_folder}{idx1}_{idx2}_corr.txt')

    return result_df


def plot_piece_pairwise_indices_corr(df: pd.DataFrame,
                                     indices_pair: Tuple[
                                         Literal["ILC"], Literal["OLC", "ILD"]],
                                     by_period: bool,
                                     corr_method: Literal["pearson", "spearman"],
                                     repo_dir: str,
                                     outliers_to_exclude_major: Optional[List[Tuple[str, str]]],
                                     outliers_to_exclude_minor: Optional[List[Tuple[str, str]]],
                                     save: bool = True):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="correlation_analyses",
                                       repo_dir=repo_dir)

    idx1, idx2 = indices_pair

    major_df = get_piece_df_by_localkey_mode(df=df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor")

    if outliers_to_exclude_major:
        major_df = exclude_piece_from_corpus(df=major_df, corpus_piece_tups=outliers_to_exclude_major)
    else:
        major_df = major_df
    if outliers_to_exclude_minor:
        minor_df = exclude_piece_from_corpus(df=minor_df, corpus_piece_tups=outliers_to_exclude_minor)
    else:
        minor_df = minor_df

    periods = Fabian_periods
    colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
    year_div = [f"<1662", f"1662-1763", f"1763-1821", f"1821-1869", f">1869"]

    # fig params:
    num_periods = len(periods)
    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols,
                            layout="constrained",
                            figsize=(3 * num_periods + 1, 6))
    # period axes:
    if by_period:
        for i, period in enumerate(periods):
            major_period_df = get_period_df(df=major_df, period=period, method="Fabian")
            minor_period_df = get_period_df(df=minor_df, period=period, method="Fabian")

            major_r, major_p, minor_r, minor_p = _global_indices_corr_by_mode(major_df=major_period_df,
                                                                              minor_df=minor_period_df,
                                                                              indices_pair=indices_pair,
                                                                              out_type="tuple",
                                                                              corr_method=corr_method)
            major_p_txt = pprint_p_text(major_p)
            minor_p_txt = pprint_p_text(minor_p)

            # plotting
            row_index = i // num_cols
            col_index = i % num_cols
            axs[row_index, col_index].set_title(year_div[i], fontweight="bold", fontsize=18, family="sans-serif")
            ma = sns.regplot(ax=axs[row_index, col_index], data=major_period_df, x=idx1, y=idx2,
                             color=colors[i], marker='o', scatter_kws={'s': 10})
            mi = sns.regplot(ax=axs[row_index + 1, col_index], data=minor_period_df, x=idx1, y=idx2,
                             color=colors[i], marker='o', scatter_kws={'s': 10})

            plt.sca(axs[row_index, col_index])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            # add stats vals in plot
            ma_x_limit = axs[row_index, col_index].get_xlim()
            mi_x_limit = axs[row_index + 1, col_index].get_xlim()

            ma_y_limit = axs[row_index, col_index].get_ylim()
            mi_y_limit = axs[row_index + 1, col_index].get_ylim()

            ma_x_pos = ma_x_limit[1] - 0.03 * (ma_x_limit[1] - ma_x_limit[0])
            mi_x_pos = mi_x_limit[1] - 0.03 * (mi_x_limit[1] - mi_x_limit[0])

            ma_y_pos = ma_y_limit[1] - 0.03 * (ma_y_limit[1] - ma_y_limit[0])
            mi_y_pos = mi_y_limit[1] - 0.03 * (mi_y_limit[1] - mi_y_limit[0])

            ma.text(ma_x_pos, ma_y_pos, f'r = {major_r}, {major_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                    va='top')
            mi.text(mi_x_pos, mi_y_pos, f'r = {minor_r}, {minor_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                    va='top')

            if i == 0:
                ma.set_ylabel("Major", fontsize=15)
                mi.set_ylabel("Minor", fontsize=15)
                ma.set_xlabel("")
                mi.set_xlabel("")

            else:
                ma.set_xlabel("")
                ma.set_ylabel("")
                mi.set_ylabel("")
                mi.set_xlabel("")

        anno = f'_byPeriod'

    else:

        major_r, major_p, minor_r, minor_p = _global_indices_corr_by_mode(major_df=major_df,
                                                                          minor_df=minor_df,
                                                                          indices_pair=indices_pair,
                                                                          out_type="tuple",
                                                                          corr_method=corr_method)
        major_p_txt = pprint_p_text(major_p)
        minor_p_txt = pprint_p_text(minor_p)

        fig, axs = plt.subplots(1, 2,
                                layout="constrained",
                                figsize=(10, 5))

        ma = sns.regplot(ax=axs[0], data=major_df, x=idx1, y=idx2,
                         marker='o', scatter_kws={'s': 10, 'alpha': 0.6}, color='#db5f57')
        mi = sns.regplot(ax=axs[1], data=minor_df, x=idx1, y=idx2,
                         marker='o', scatter_kws={'s': 10, 'alpha': 0.6}, color='#39a7d0')

        # add stats vals in plot
        ma_x_limit = axs[0].get_xlim()
        mi_x_limit = axs[1].get_xlim()

        ma_y_limit = axs[0].get_ylim()
        mi_y_limit = axs[1].get_ylim()

        ma_x_pos = ma_x_limit[1] - 0.03 * (ma_x_limit[1] - ma_x_limit[0])
        mi_x_pos = mi_x_limit[1] - 0.03 * (mi_x_limit[1] - mi_x_limit[0])

        ma_y_pos = ma_y_limit[1] - 0.03 * (ma_y_limit[1] - ma_y_limit[0])
        mi_y_pos = mi_y_limit[1] - 0.03 * (mi_y_limit[1] - mi_y_limit[0])

        ma.text(ma_x_pos, ma_y_pos, f'r = {major_r}, {major_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                va='top')
        mi.text(mi_x_pos, mi_y_pos, f'r = {minor_r}, {minor_p_txt}', fontsize=13, fontstyle='italic', ha='right',
                va='top')

        ma.set_ylabel("")
        mi.set_ylabel("")
        ma.set_xlabel("Major", fontsize=11)
        mi.set_xlabel("Minor", fontsize=11)
        anno = ""

    fig.supxlabel("\nIn-Label Chromaticity (ILC)", fontsize=15, fontdict=dict(weight='bold'))
    if idx2 == "OLC":
        fig.supylabel("Out-of-Label Chromaticity (OLC)\n", fontsize=15, fontdict=dict(weight='bold'))
    elif idx2 == "ILD":
        fig.supylabel("In-Label Dissonance (ILD)\n", fontsize=15, fontdict=dict(weight='bold'))
    else:
        raise ValueError

    # save plot
    if save:
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        plt.savefig(f'{sub_corr_analysis_folder}{idx1}_{idx2}_corr{anno}.pdf', dpi=200)


def plotly_piece_pairwise_indices_faceting(df: pd.DataFrame, indices_pair: Tuple[
    Literal["ILC"], Literal["OLC", "ILD"]], save: bool, repo_dir: str):
    """
        df: assuming we take the piece_level_indices_by_mode df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="correlation_analyses",
                                       repo_dir=repo_dir)

    idx1, idx2 = indices_pair
    fig = px.scatter(df, x=idx1, y=idx2, color="corpus", facet_col="period", facet_row="localkey_mode",
                     hover_data="piece")
    if save:
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        fig.write_html(f'{sub_corr_analysis_folder}plotly_{idx1}_{idx2}_scatter.html')


# %% Analysis: stats by corpus


def _barplot_indices_by_corpus(df: pd.DataFrame, repo_dir: str):
    """
    df: assume taking the corpora level indices df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 10), layout="constrained", sharey=True)

    df["corpus_pp"] = df["corpus"].map(corpus_prettyprint_dict)

    bar_width = 0.75
    # major_color = '#95b5d3'
    # minor_color = '#4b698f'

    major_color = '#e08791'  # red
    minor_color = '#92abc5'  # blue

    sns.barplot(ax=axs[0], data=df, x="ILC", y="corpus_pp", hue="localkey_mode", errorbar="ci", legend='brief',
                palette=[major_color, minor_color], width=bar_width, saturation=1)
    sns.barplot(ax=axs[1], data=df, x="OLC", y="corpus_pp", hue="localkey_mode", errorbar="ci", legend=False,
                palette=[major_color, minor_color], width=bar_width, saturation=1)
    sns.barplot(ax=axs[2], data=df, x="ILD", y="corpus_pp", hue="localkey_mode", errorbar="ci", legend=False,
                palette=[major_color, minor_color], width=bar_width, saturation=1)

    axs[0].set_ylabel("")
    axs[0].legend(loc="upper right", bbox_to_anchor=(-0.25, 1.02))
    axs[0].set_xlabel("ILC", fontweight="bold", fontsize=15)
    axs[1].set_xlabel("OLC", fontweight="bold", fontsize=15)
    axs[2].set_xlabel("ILD", fontweight="bold", fontsize=15)

    axs[2].set_xlim(left=0.2)

    for x in range(3):
        axs[x].spines['top'].set_visible(False)
        axs[x].spines['right'].set_visible(False)
        axs[x].spines['bottom'].set_visible(True)
        axs[x].spines['left'].set_visible(True)

    fig.supylabel("Corpora", fontweight="bold", fontsize=16)

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}barplot_corpora_indices.pdf', dpi=200)


def _barplot_chromaticity_by_corpus(df: pd.DataFrame, repo_dir: str):
    """
    df: assume taking the corpora level indices df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    fig, axs = plt.subplots(ncols=2, figsize=(15, 10), layout="constrained", sharey=True)

    df["corpus_pp"] = df["corpus"].map(corpus_prettyprint_dict)

    bar_width = 0.6
    wlc_color = "#75c1c4"  # green
    olc_color = '#b2b6b6'  # gray

    # olc_color = "#7e598d" # purple
    # wlc_color = '#eab00e'   # yellow

    # wlc_color = "#e3705e" # red
    # olc_color = '#01a7ac'   # green
    #
    # wlc_color = "#e7cd79" # yellow
    # olc_color = '#467897'   # blue

    palette = [wlc_color, olc_color]

    major_df = get_piece_df_by_localkey_mode(df=df, mode="major").sort_values(["corpus_id"])
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor").sort_values(["corpus_id"])

    major_longform = pd.melt(major_df, id_vars=["corpus_pp", "corpus_id"], value_vars=["ILC", "OLC"],
                             var_name="index_type", value_name="value")
    minor_longform = pd.melt(minor_df, id_vars=["corpus_pp", "corpus_id"], value_vars=["ILC", "OLC"],
                             var_name="index_type", value_name="value")

    sns.barplot(ax=axs[0], data=major_longform, x="value", y="corpus_pp", hue="index_type", errorbar=None,
                legend='brief',
                palette=palette, width=bar_width)
    sns.barplot(ax=axs[1], data=minor_longform, x="value", y="corpus_pp", hue="index_type", errorbar=None, legend=False,
                palette=palette, width=bar_width)
    # sns.barplot(ax=axs[2], data=longform_df, x="ILD", y="corpus_pp", hue="localkey_mode", errorbar=None, legend=False,
    #             palette=[major_color, minor_color], width=bar_width)

    axs[0].set_ylabel("")
    axs[0].legend(loc="upper right", bbox_to_anchor=(-0.25, 1.02))
    # axs[0].set_xlabel("Major", fontweight="bold", fontsize=15)
    axs[0].set_title("Major", fontweight="bold", fontsize=15)
    axs[1].set_title("Minor", fontweight="bold", fontsize=15)
    # axs[1].set_xlabel("Minor", fontweight="bold", fontsize=15)
    # axs[2].set_xlabel("ILD", fontweight="bold", fontsize=15)
    #
    # axs[2].set_xlim(left=0.2)

    for x in range(2):
        axs[x].spines['top'].set_visible(False)
        axs[x].spines['right'].set_visible(False)
        axs[x].spines['bottom'].set_visible(True)
        axs[x].spines['left'].set_visible(True)

    fig.supylabel("Corpora", fontweight="bold", fontsize=16)

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}barplot_corpora_chromaticity_GreenGray.pdf', dpi=200)


def barplot_dissonance_by_corpus(df: pd.DataFrame, mode: Literal["major", "minor"], repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")

    df["corpus_pp"] = df["corpus"].map(corpus_prettyprint_dict)
    bar_width = 0.6

    major_color = '#95b5d3'
    minor_color = '#4b698f'
    palette = [major_color, minor_color]
    sns.barplot(ax=ax, data=df, x="corpus_pp", y="ILD", hue="localkey_mode", errorbar=None, legend='brief',
                palette=palette, width=bar_width)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylim(0, 6)
    ax.set_ylabel("ILD", fontweight="bold", fontsize=13)
    ax.set_xlabel("Corpus", fontweight="bold", fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(frameon=False, ncol=2)

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}barplot_corpora_dissonance.pdf', dpi=200)


def barplot_indices_by_corpus(df: pd.DataFrame, mode: Literal["major", "minor"], repo_dir: str,
                              anno: Optional[str] = None):
    """
    df: assume taking the corpora level indices df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode).sort_values(["corpus_id"])

    fig, axs = plt.subplots(nrows=3, figsize=(10, 15), layout="constrained", sharex=True)

    mode_df["corpus_pp"] = mode_df["corpus"].map(corpus_prettyprint_dict)

    bar_width = 0.75

    if mode == "major":
        color = 'lightcoral'  # red
    else:
        color = 'cornflowerblue'  # blue

    sns.barplot(ax=axs[0], data=mode_df, x="corpus_id", y="ILC",
                color=color, width=bar_width)
    sns.barplot(ax=axs[1], data=mode_df, x="corpus_id", y="OLC",
                color=color, width=bar_width)
    sns.barplot(ax=axs[2], data=mode_df, x="corpus_id", y="ILD",
                color=color, width=bar_width)

    # axs[0].bar_label(axs[0].containers[0], fontsize=5, rotation=90)
    # axs[1].bar_label(axs[1].containers[0], fontsize=5, rotation=90)
    # axs[2].bar_label(axs[2].containers[0], fontsize=5, rotation=90)

    axs[0].set_ylabel("ILC", fontweight="bold", fontsize=15)
    axs[1].set_ylabel("OLC", fontweight="bold", fontsize=15)
    axs[2].set_ylabel("ILD", fontweight="bold", fontsize=15)
    axs[2].set_xlabel("")

    axs[2].set_xticks(range(len(mode_df)), labels=mode_df["corpus_pp"])
    axs[2].tick_params(axis='x', labelrotation=90)

    for x in range(3):
        axs[x].spines['top'].set_visible(False)
        axs[x].spines['right'].set_visible(False)
        axs[x].spines['bottom'].set_visible(True)
        axs[x].spines['left'].set_visible(True)

    # fig.supxlabel("Corpora", fontweight="bold", fontsize=16)

    if anno:
        txt = f'_{anno}'
    else:
        txt = ""

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}barplot_corpora_indices_{mode}{txt}.pdf', dpi=200)


# %% Analysis: piece analysis: chord-level chromaticity and dissonance across piece


def get_piece_df(df: pd.DataFrame, corpus: str | int, piece: str | int, save: bool | str = False) -> pd.DataFrame:
    if isinstance(corpus, str):
        piece_df = df.loc[(df['corpus'] == corpus) & (df['piece'] == piece)]
    elif isinstance(corpus, int):
        piece_df = df.loc[(df['corpus_id'] == corpus) & (df['piece_id'] == piece)]
    else:
        raise NotImplementedError
    piece_df.loc[:, "quarterbeats"] = piece_df["quarterbeats"].apply(lambda x: float(Fraction(x)))
    piece_df = piece_df.sort_values(["quarterbeats"])
    c = piece_df["corpus"].tolist()[0]
    p = piece_df["piece"].tolist()[0]
    if save:
        assert isinstance(save, str)
        if not os.path.exists(save):
            os.makedirs(save)
        corpus_piece_name = f'{c}_{p}.tsv'
        piece_df.to_csv(f'{save}{corpus_piece_name}', sep='\t')
    return piece_df


def plot_chord_ilc_ild_in_piece(df: pd.DataFrame):
    # Convert quarterbeats to float
    df["quarterbeats"] = df["quarterbeats"].apply(lambda x: float(Fraction(x)))

    # Plot ILC and ILD over quarterbeats
    plt.figure(figsize=(20, 4))

    # Plot ILC and ILD as line plots for a smooth time series appearance
    plt.plot(df["quarterbeats"], df["ILC"], 'o-', label="ILC", color="lightseagreen")
    plt.plot(df["quarterbeats"], df["ILD"], 'x-', label="ILD", color="sienna")

    # Shade regions where mode is minor, avoiding re-shading already shaded regions
    in_minor_section = False
    start = None

    for i, row in df.iterrows():
        if row["localkey_mode"] == "minor" and not in_minor_section:
            # Start of a new minor section
            start = row["quarterbeats"]
            in_minor_section = True
        elif row["localkey_mode"] != "minor" and in_minor_section:
            # End of the current minor section
            end = row["quarterbeats"]
            plt.axvspan(start, end, color='gray', alpha=0.2, edgecolor='none')
            in_minor_section = False

    # Ensure the last minor section gets shaded if it runs to the end
    if in_minor_section:
        plt.axvspan(start, df["quarterbeats"].iloc[-1], color='gray', alpha=0.2, edgecolor='none')

    corpus = df["corpus"].tolist()[0]
    piece = df["piece"].tolist()[0]
    # Labeling
    plt.xlabel("Chord")
    plt.ylabel("Values")
    plt.title(f"{corpus} : {piece}")
    plt.legend()
    plt.xticks([], [])
    plt.xlim(-0.5, df["quarterbeats"].iloc[-1] + 0.5)
    plt.tight_layout()
    plt.show()



def plot_schumann_ilc_ild_in_piece(df: pd.DataFrame, normalize_method: Literal["standardize", "min-max"], repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="piece_fig_ilc_ild", repo_dir=repo_dir)

    # Convert quarterbeats to float
    df["quarterbeats"] = df["quarterbeats"].apply(lambda x: float(Fraction(x)))

    if normalize_method == "standardize":
        wo_repeat_df = (
            df[(df['quarterbeats'] <= 1.0) | (df['quarterbeats'] >= 34)]
            .assign(quarterbeats=lambda x: x['quarterbeats'].replace(
                {df['quarterbeats'].iloc[0]: 32, df['quarterbeats'].iloc[1]: 33}))
            .assign(quarterbeats=lambda x: x['quarterbeats'] - x['quarterbeats'].min())
            .assign(
                ILC=lambda x: (x['ILC'] - x['ILC'].mean()) / x['ILC'].std(),
                ILD=lambda x: (x['ILD'] - x['ILD'].mean()) / x['ILD'].std()
            )
        )
    elif normalize_method == "min-max":
        wo_repeat_df = (
            df[(df['quarterbeats'] <= 1.0) | (df['quarterbeats'] >= 34)]
            .assign(quarterbeats=lambda x: x['quarterbeats'].replace(
                {df['quarterbeats'].iloc[0]: 32, df['quarterbeats'].iloc[1]: 33}))
            .assign(quarterbeats=lambda x: x['quarterbeats'] - x['quarterbeats'].min())
            .assign(
                ILC=lambda x: (x['ILC'] - x['ILC'].min()) / (x['ILC'].max() - x['ILC'].min()),
                ILD=lambda x: (x['ILD'] - x['ILD'].min()) / (x['ILD'].max() - x['ILD'].min())
            )
        )
    else:
        raise NotImplementedError
    # Plot ILC and ILD over quarterbeats
    fig, ax = plt.subplots(figsize=(20, 4))

    # Plot ILC and ILD as line plots for a smooth time series appearance
    plt.plot(wo_repeat_df["quarterbeats"], wo_repeat_df["ILC"], 'o-', label="ILC", color="lightseagreen")
    plt.plot(wo_repeat_df["quarterbeats"], wo_repeat_df["ILD"], 'x-', label="ILD", color="sienna")

    # Shade regions where mode is minor, avoiding re-shading already shaded regions

    if normalize_method == "standardize":
        # Set the maximum height of the shaded region
        max_y_value = max(wo_repeat_df["ILC"].max(),wo_repeat_df["ILD"].max())
        min_y_value = min(wo_repeat_df["ILC"].min(),wo_repeat_df["ILD"].min())
        in_minor_section = False
        start = None

        for i, row in wo_repeat_df.iterrows():
            if row["localkey_mode"] == "minor" and not in_minor_section:
                # Start of a new minor section
                start = row["quarterbeats"]
                in_minor_section = True
            elif row["localkey_mode"] != "minor" and in_minor_section:
                # End of the current minor section
                end = row["quarterbeats"]
                plt.fill_between([start, end], min_y_value-0.05, max_y_value+0.05, color='gray', alpha=0.2)
                in_minor_section = False

        # Ensure the last minor section gets shaded if it runs to the end
        if in_minor_section:
            plt.fill_between([start, wo_repeat_df["quarterbeats"].iloc[-1]], min_y_value-0.05, max_y_value+0.05, color='gray', alpha=0.2)

    # if normalize_method == "standardize":
    #     for i, row in wo_repeat_df.iterrows():
    #         if row["localkey_mode"] == "minor" and not in_minor_section:
    #             # Start of a new minor section
    #             start = row["quarterbeats"]
    #             in_minor_section = True
    #         elif row["localkey_mode"] != "minor" and in_minor_section:
    #             # End of the current minor section
    #             end = row["quarterbeats"]
    #             plt.axvspan(start, end, color='gray', alpha=0.2)
    #             in_minor_section = False
    #
    #     # Ensure the last minor section gets shaded if it runs to the end
    #     if in_minor_section:
    #         plt.axvspan(start, wo_repeat_df["quarterbeats"].iloc[-1], color='gray', alpha=0.2)

    elif normalize_method == "min-max":
        in_minor_section = False
        start = None
        for i, row in wo_repeat_df.iterrows():
            if row["localkey_mode"] == "minor" and not in_minor_section:
                # Start of a new minor section
                start = row["quarterbeats"]
                in_minor_section = True
            elif row["localkey_mode"] != "minor" and in_minor_section:
                # End of the current minor section
                end = row["quarterbeats"]
                # Use fill_between to shade between start and end, limiting y-values
                plt.fill_between([start, end], y1=-0.05, y2=1.05, color='gray', alpha=0.2)
                in_minor_section = False
        ax.set_ylim(-0.1, 1.15)

        # Ensure the last minor section gets shaded if it runs to the end
        if in_minor_section:
            plt.fill_between([start, wo_repeat_df["quarterbeats"].iloc[-1]], y1=-0.05, y2=1.05, color='gray', alpha=0.2)
    else:
        raise NotImplementedError
    corpus = wo_repeat_df["corpus"].tolist()[0]
    piece = wo_repeat_df["piece"].tolist()[0]
    # Labeling
    xticks = wo_repeat_df["quarterbeats"].tolist()  # Get the positions for the ticks
    xtickslabels = wo_repeat_df["chord"].tolist()

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels, rotation='vertical', fontsize=10)

    ax.set_xlabel("Chord", fontweight="bold", fontsize="large")
    ax.set_ylabel("Values", fontweight="bold", fontsize="large")
    ax.set_xlim(-0.5, wo_repeat_df["quarterbeats"].iloc[-1] + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add repeat signs
    repeatL = imread(f"{repo_dir}/Results/piece_fig_ilc_ild/repeat_left.png")
    repeatR = imread(f"{repo_dir}/Results/piece_fig_ilc_ild/repeat_right.png")
    endsign = imread(f"{repo_dir}/Results/piece_fig_ilc_ild/end_sign.png")
    imagebox1 = OffsetImage(repeatL, zoom=0.04)
    imagebox2 = OffsetImage(repeatR, zoom=0.04)
    imagebox3 = OffsetImage(endsign, zoom=0.04)

    _, y_max = plt.gca().get_ylim()
    # L = AnnotationBbox(imagebox1, (2, 2.1), frameon=False)
    L = AnnotationBbox(imagebox1, (2, y_max-0.03), frameon=False)
    R = AnnotationBbox(imagebox2, (34, y_max-0.03), frameon=False)
    E = AnnotationBbox(imagebox3, (44, y_max-0.03), frameon=False)
    ax.add_artist(L)
    ax.add_artist(R)
    ax.add_artist(E)
    # legend
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # compute r vals
    major_df = wo_repeat_df[wo_repeat_df['localkey_mode'] == 'major']
    minor_df = wo_repeat_df[wo_repeat_df['localkey_mode'] == 'minor']

    r_major = major_df[["ILC", "ILD"]].corr(method='pearson').iloc[0, 1]
    r_minor = minor_df[["ILD", "ILC"]].corr(method='pearson').iloc[0, 1]

    legend_item1 = Line2D([0], [0], color='none', label=r"$r_{\text{major}} = " + f"{r_major:.2f}$")
    legend_item2 = Line2D([0], [0], color='none', label=r"$r_{\text{minor}} = " + f"{r_minor:.2f}$")
    legend_items = current_handles + [legend_item1, legend_item2]

    # Add the custom legend item to the plot
    # ax.legend(handles=legend_items, loc='outside upper right', fontsize='small')
    fig.legend(handles=legend_items, bbox_to_anchor=(0.97, 0.88), loc='upper right', fontsize='small')

    plt.tight_layout()

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}{corpus}_{piece}.pdf', dpi=200)
    plt.show()


def plot_ilc_ild_in_piece(df: pd.DataFrame, normalize_method: Literal["standardize", "min-max"], legend_pos: Tuple[float, float], repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="piece_fig_ilc_ild", repo_dir=repo_dir)

    # Convert quarterbeats to float
    df["quarterbeats"] = df["quarterbeats"].apply(lambda x: float(Fraction(x)))

    if normalize_method == "standardize":
        df = df.assign(
            ILC=lambda x: (x['ILC'] - x['ILC'].mean()) / x['ILC'].std(),
            ILD=lambda x: (x['ILD'] - x['ILD'].mean()) / x['ILD'].std()
        )
    elif normalize_method == "min-max":
        df = df.assign(
            ILC=lambda x: (x['ILC'] - x['ILC'].min()) / (x['ILC'].max() - x['ILC'].min()),
            ILD=lambda x: (x['ILD'] - x['ILD'].min()) / (x['ILD'].max() - x['ILD'].min())
        )
    else:
        raise NotImplementedError
    # Plot ILC and ILD over quarterbeats
    fig, ax = plt.subplots(figsize=(20, 4))

    # Plot ILC and ILD as line plots for a smooth time series appearance
    plt.plot(df["quarterbeats"], df["ILC"], 'o-', label="ILC", color="lightseagreen")
    plt.plot(df["quarterbeats"], df["ILD"], 'x-', label="ILD", color="sienna")

    # Shade regions where mode is minor, using fill_between for custom height
    if normalize_method == "standardize":
        # Set the maximum height of the shaded region
        max_y_value = max(df["ILC"].max(),df["ILD"].max())
        min_y_value = min(df["ILC"].min(),df["ILD"].min())
        in_minor_section = False
        start = None

        for i, row in df.iterrows():
            if row["localkey_mode"] == "minor" and not in_minor_section:
                # Start of a new minor section
                start = row["quarterbeats"]
                in_minor_section = True
            elif row["localkey_mode"] != "minor" and in_minor_section:
                # End of the current minor section
                end = row["quarterbeats"]
                plt.fill_between([start, end], min_y_value-0.05, max_y_value+0.05, color='gray', alpha=0.2)
                in_minor_section = False

        # Ensure the last minor section gets shaded if it runs to the end
        if in_minor_section:
            plt.fill_between([start, df["quarterbeats"].iloc[-1]], min_y_value-0.05, max_y_value+0.05, color='gray', alpha=0.2)

    # in_minor_section = False
    # start = None
    #
    # if normalize_method=="min-max":
    #     for i, row in df.iterrows():
    #         if row["localkey_mode"] == "minor" and not in_minor_section:
    #             # Start of a new minor section
    #             start = row["quarterbeats"]
    #             in_minor_section = True
    #         elif row["localkey_mode"] != "minor" and in_minor_section:
    #             # End of the current minor section
    #             end = row["quarterbeats"]
    #             plt.fill_between([start, end], -0.05, 1.05, color='gray', alpha=0.2)
    #             in_minor_section = False
    #
    #     # Ensure the last minor section gets shaded if it runs to the end
    #     if in_minor_section:
    #         plt.fill_between([start, df["quarterbeats"].iloc[-1]], -0.05, 1.05, color='gray', alpha=0.2)
    else:
        in_minor_section = False
        start = None

        for i, row in df.iterrows():
            if row["localkey_mode"] == "minor" and not in_minor_section:
                # Start of a new minor section
                start = row["quarterbeats"]
                in_minor_section = True
            elif row["localkey_mode"] != "minor" and in_minor_section:
                # End of the current minor section
                end = row["quarterbeats"]
                plt.axvspan(start, end, color='gray', alpha=0.2)
                in_minor_section = False
        ax.set_ylim(-0.1, 1.1)

        # Ensure the last minor section gets shaded if it runs to the end
        if in_minor_section:
            plt.axvspan(start, df["quarterbeats"].iloc[-1], color='gray', alpha=0.2)

    corpus = df["corpus"].tolist()[0]
    piece = df["piece"].tolist()[0]
    # Labeling
    xticks = df["quarterbeats"].tolist()  # Get the positions for the ticks
    xtickslabels = df["chord"].tolist()

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels, rotation='vertical', fontsize=10)

    ax.set_xlabel("Chord")
    ax.set_ylabel("Values")
    ax.set_xlim(-0.5, df["quarterbeats"].iloc[-1] + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # compute r vals
    major_df = df[df['localkey_mode'] == 'major']
    minor_df = df[df['localkey_mode'] == 'minor']

    r_major = major_df[["ILC", "ILD"]].corr(method='pearson').iloc[0, 1]
    r_minor = minor_df[["ILD", "ILC"]].corr(method='pearson').iloc[0, 1]

    # Define a custom legend item with just the text, no line.
    legend_item1 = Line2D([0], [0], color='none', label=r"$r_{\text{major}} = " + f"{r_major:.2f}$")
    legend_item2 = Line2D([0], [0], color='none', label=r"$r_{\text{minor}} = " + f"{r_minor:.2f}$")
    legend_items = current_handles + [legend_item1, legend_item2]

    # Add the custom legend item to the plot
    plt.legend(handles=legend_items, bbox_to_anchor=legend_pos, loc='upper right', fontsize='small')

    plt.tight_layout()

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}{corpus}_{piece}.pdf', dpi=200)
    plt.show()


def stats_r_vals_ilc_ild(df:pd.DataFrame, repo_dir: str):

    major_df = df[df['localkey_mode'] == 'major']
    minor_df = df[df['localkey_mode'] == 'minor']
    major_seg_num = major_df.shape[0]
    minor_seg_num = minor_df.shape[0]
    # r=0
    r0 = df[df['r_ilc_ild'] == 0]
    r0_major = major_df[major_df['r_ilc_ild'] == 0]
    r0_minor = minor_df[minor_df['r_ilc_ild'] == 0]

    r_pos = df[df['r_ilc_ild'] > 0]
    r_pos_major = major_df[major_df['r_ilc_ild'] > 0]
    r_pos_minor = minor_df[minor_df['r_ilc_ild'] > 0]

    r_neg = df[df['r_ilc_ild'] < 0]
    r_neg_major = major_df[major_df['r_ilc_ild'] < 0]
    r_neg_minor = minor_df[minor_df['r_ilc_ild'] < 0]

    res_dict = {
                "total seg num:": df.shape[0],
                "major seg num:": major_seg_num,
                "minor seg num:": minor_seg_num,

                "r=0 num:": r0.shape[0],
                "r=0 (major) num:": r0_major.shape[0],
                "r=0 (minor) num:": r0_minor.shape[0],

                "r>0 num:": r_pos.shape[0],
                "r>0 (major) num:": r_pos_major.shape[0],
                "r>0 (minor) num:": r_pos_minor.shape[0],

                "r<0 num:": r_neg.shape[0],
                "r<0 (major) num:": r_neg_major.shape[0],
                "r<0 (minor) num:": r_neg_minor.shape[0],
                }
    print(res_dict)




# %% full analyses set for the paper:

def full_analyses_set_for_paper():
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'

    # Loading relevant dfs: _____________________________
    print(f'Loading dfs ...')
    prep_DLC = load_file_as_df(f"{repo}Data/prep_data/processed_DLC_data.pickle")

    chord_level_indices = load_file_as_df(
        f"{repo}Data/prep_data/for_analysis/chord_level_indices.pickle")

    chord_indices_r_vals_by_piece = load_file_as_df(
        f"{repo}Data/prep_data/for_analysis/chord_indices_r_vals_by_piece.pickle")

    piece_level_indices_by_mode = load_file_as_df(
        f"{repo}Data/prep_data/for_analysis/piece_level_indices_by_mode.pickle")

    chromaticity_piece_by_mode = load_file_as_df(
        f"{repo}Data/prep_data/for_analysis/chromaticity_piece_by_mode.pickle")

    corpora_indices_by_mode = load_file_as_df(f"{repo}Data/prep_data/for_analysis/corpora_level_indices_by_mode.pickle")

    # Analysis: Basic stats and plots _________________________________________________________

    print(f'DLC corpus basic stats ...')
    DLC_corpus_stats(chord_level_df=prep_DLC, repo_dir=repo)

    print(f'Analysis: Basic stats for chromaticity and dissonance ...')
    stats_by_piece(df=chord_level_indices, repo_dir=repo)
    stats_by_corpus(df=chord_level_indices, repo_dir=repo)

    # Analysis: Piece distributions fig and table _________________________________________________________
    print(f'Analysis: Piece distributions fig and table ...')

    piece_distribution(df=prep_DLC, repo_dir=repo)

    # Correlation Analysis: piece-level ILC and OLC ____________________________________
    print(f'Correlation Analysis: piece-level chromaticities ...')

    # manually remove outliers -------:

    ## current criteria: OLC >10 OR ILC >10
    ILC_OLC_major_outliers = [("chopin_mazurkas", "BI162-3op63-3"),
                              ("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca"),
                              ("tchaikovsky_seasons", "op37a12"), ("dvorak_silhouettes", "op08n12"),
                              ("dvorak_silhouettes", "op08n01")]

    ILC_OLC_minor_outliers = [("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca"),
                              ("bartok_bagatelles", "op06n12")]  # bartok_bagatelles: ILC>10
    # end of outlier list -------------

    compute_piece_corr_stats(df=piece_level_indices_by_mode, indices_pair=("ILC", "OLC"),
                             repo_dir=repo, corr_method="pearson",
                             outliers_to_exclude_major=ILC_OLC_major_outliers,
                             outliers_to_exclude_minor=ILC_OLC_minor_outliers,
                             save=True)

    plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("ILC", "OLC"),
                                     repo_dir=repo, corr_method="pearson",
                                     by_period=False,
                                     outliers_to_exclude_major=ILC_OLC_major_outliers,
                                     outliers_to_exclude_minor=ILC_OLC_minor_outliers,
                                     save=True)

    plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("ILC", "OLC"),
                                     repo_dir=repo, corr_method="pearson",
                                     by_period=True,
                                     outliers_to_exclude_major=ILC_OLC_major_outliers,
                                     outliers_to_exclude_minor=ILC_OLC_minor_outliers,
                                     save=True)

    plotly_piece_pairwise_indices_faceting(df=piece_level_indices_by_mode,
                                           indices_pair=("ILC", "OLC"), repo_dir=repo, save=True)

    # Correlation Analysis: piece-level ILC and ILD ____________________________________
    print(f'Correlation Analysis: piece-level chromaticity and dissonance ...')

    # manually remove outliers -------:
    ILC_ILD_major_outliers = None
    ILC_ILD_minor_outliers = [("bartok_bagatelles", "op06n12")]  # bartok_bagatelles: ILC>10

    compute_piece_corr_stats(df=piece_level_indices_by_mode, indices_pair=("ILC", "ILD"),
                             repo_dir=repo, corr_method="pearson",
                             outliers_to_exclude_major=ILC_ILD_major_outliers,
                             outliers_to_exclude_minor=ILC_ILD_minor_outliers,
                             save=True)

    plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("ILC", "ILD"),
                                     repo_dir=repo, corr_method="pearson", by_period=False,
                                     outliers_to_exclude_major=ILC_ILD_major_outliers,
                                     outliers_to_exclude_minor=ILC_ILD_minor_outliers,
                                     save=True)
    plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("ILC", "ILD"),
                                     repo_dir=repo, corr_method="pearson", by_period=True,
                                     outliers_to_exclude_major=ILC_ILD_major_outliers,
                                     outliers_to_exclude_minor=ILC_ILD_minor_outliers,
                                     save=True)

    plotly_piece_pairwise_indices_faceting(df=piece_level_indices_by_mode,
                                           indices_pair=("ILC", "ILD"), repo_dir=repo, save=True)

    # Correlation Analysis: chord-indices r-vals in pieces _____________________________
    print(f'Anallysis: Chord-level indices correlations (r-vals) in pieces...')

    # Analysis: Corpora-level indices:

    # barplot_dissonance_by_corpus(df=corpora_indices_by_mode, repo_dir=repo)
    barplot_indices_by_corpus(df=corpora_indices_by_mode, repo_dir=repo, mode="major")
    barplot_indices_by_corpus(df=corpora_indices_by_mode, repo_dir=repo, mode="minor")

    print(f'Fini!')


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'

    # r_df = load_file_as_df(path=f'{repo}Data/prep_data/for_analysis/chord_indices_r_vals_by_piece.pickle')
    # stats_r_vals_ilc_ild(r_df, repo_dir=repo)

    # assert False
    df = load_file_as_df(path=f"{repo}Data/prep_data/for_analysis/chord_level_indices.pickle")

    test = get_piece_df(df=df, corpus="bartok_bagatelles", piece="op06n01", save=f'{repo}Data/prep_data/piece_dfs/')
    plot_ilc_ild_in_piece(df=test, repo_dir=repo, normalize_method="standardize", legend_pos=(0.97, 0.98))


    # grieg = get_piece_df(df=df, corpus="grieg_lyric_pieces", piece="op68n06", save=f'{repo}Data/prep_data/piece_dfs/')
    # bach_1009 = get_piece_df(df=df, corpus="bach_solo", piece="BWV1009_06_BourréeII",
    #                          save=f'{repo}Data/prep_data/piece_dfs/')
    #
    # # plot_ilc_ild_in_piece(df=bach_1009, repo_dir=repo, normalize_method="min-max", legend_pos=(0.26, 0.93))
    # plot_ilc_ild_in_piece(df=grieg, repo_dir=repo, normalize_method="standardize", legend_pos=(0.97, 0.98))
    #
    assert False
    shumann = get_piece_df(df=df, corpus=25, piece=8, save=f'{repo}Data/prep_data/piece_dfs/')
    schuman_kind = get_piece_df(df=df, corpus="schumann_kinderszenen", piece="n03")
    plot_ilc_ild_in_piece(df=schuman_kind, repo_dir=repo, normalize_method="standardize", legend_pos=(0.2,0.9))
