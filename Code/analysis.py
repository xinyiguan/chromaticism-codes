import fractions
import json
import os
from typing import Literal, Optional, Tuple, List
import pandas as pd
import seaborn as sns
import pingouin as pg
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from Code.utils.util import load_file_as_df, corpus_composer_dict, corpus_collection_dict, corpus_prettyprint_dict, \
    save_df
from Code.utils.auxiliary import create_results_folder, determine_period_id, get_period_df, Johannes_periods, \
    Fabian_periods, pprint_p_text, color_palette5, color_palette4, determine_group, rand_jitter, \
    get_piece_df_by_localkey_mode, exclude_piece_from_corpus
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
        period_Fabian=("period_Fabian", "first"),
        period_Johannes=("period_Johannes", "first"),

        min_WLC=("WLC", 'min'),
        avg_WLC=("WLC", 'mean'),
        max_WLC=("WLC", 'max'),

        min_OLC=("OLC", 'min'),
        avg_OLC=("OLC", 'mean'),
        max_OLC=("OLC", 'max'),

        min_WLD=("WLD", 'min'),
        avg_WLD=("WLD", 'mean'),
        max_WLD=("WLD", 'max')
    )

    pieces_df.to_latex(f'{result_dir}piece_level_indices_stats.txt')

    return pieces_df


def stats_by_piece_by_mode(piece_level_indices_df: pd.DataFrame, repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="basic_stats",
                                       repo_dir=repo_dir)

    major_df = get_piece_df_by_localkey_mode(df=piece_level_indices_df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=piece_level_indices_df, mode="minor")

    WLC_major = major_df["WLC"].mean()
    WLC_minor = minor_df["WLC"].mean()
    OLC_major = major_df["OLC"].mean()
    OLC_minor = minor_df["OLC"].mean()
    WLD_major = major_df["WLD"].mean()
    WLD_minor = minor_df["WLD"].mean()

    data = {
        "Index": ["WLC", "WLC", "OLC", "OLC", "WLD", "WLD"],
        "mode": ["major", "minor", "major", "minor", "major", "minor"],
        "value": [WLC_major, WLC_minor, OLC_major, OLC_minor, WLD_major, WLD_minor]
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

    pieces_df = df.groupby(['corpus'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),

        min_WLC=("WLC", 'min'),
        avg_WLC=("WLC", 'mean'),
        max_WLC=("WLC", 'max'),

        min_OLC=("OLC", 'min'),
        avg_OLC=("OLC", 'mean'),
        max_OLC=("OLC", 'max'),

        min_WLD=("WLD", 'min'),
        avg_WLD=("WLD", 'mean'),
        max_WLD=("WLD", 'max')
    )

    pieces_df.to_latex(f'{result_dir}corpora_level_indices_stats.txt')

    return pieces_df


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


# %% Analysis: chromaticity distributions

def _ax_chrom_distribution(ax: Axes,
                           df: pd.DataFrame,
                           logscale: bool,
                           mode: Literal["major", "minor"],
                           chromaticity_type: Literal["WLC", "OLC"]) -> Axes:
    mode_df = df.loc[(df['localkey_mode'] == mode)]
    # chrom_vals = mode_df[chromaticity_type]
    sns.histplot(ax=ax, data=mode_df, x=chromaticity_type, log_scale=logscale)
    if logscale:
        title_prefix = f"log "
    else:
        title_prefix = ""

    ax.set_title(f"{title_prefix}{chromaticity_type}({mode})", x=0.85, y=0.9)
    ax.set_xlabel("")
    return ax


def plot_chrom_distribution(df: pd.DataFrame,
                            repo_dir: str) -> None:
    """
    df: take the chromaticity_piece_by_mode df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="chrom_distributions", repo_dir=repo_dir)

    fig, axs = plt.subplots(4, 2, figsize=(12, 15),
                            layout="constrained")
    _ax_chrom_distribution(ax=axs[0, 0], df=df, mode="major", chromaticity_type="WLC", logscale=False)
    _ax_chrom_distribution(ax=axs[0, 1], df=df, mode="major", chromaticity_type="WLC", logscale=True)
    _ax_chrom_distribution(ax=axs[1, 0], df=df, mode="minor", chromaticity_type="WLC", logscale=False)
    _ax_chrom_distribution(ax=axs[1, 1], df=df, mode="minor", chromaticity_type="WLC", logscale=True)
    _ax_chrom_distribution(ax=axs[2, 0], df=df, mode="major", chromaticity_type="OLC", logscale=False)
    _ax_chrom_distribution(ax=axs[2, 1], df=df, mode="major", chromaticity_type="OLC", logscale=True)
    _ax_chrom_distribution(ax=axs[3, 0], df=df, mode="minor", chromaticity_type="OLC", logscale=False)
    _ax_chrom_distribution(ax=axs[3, 1], df=df, mode="minor", chromaticity_type="OLC", logscale=True)

    plt.savefig(f'{result_dir}chromaticity_dist_before_after_log_trans.pdf', dpi=200)


# %% Analysis: Piece distributions fig and table
def piece_distribution(df: pd.DataFrame, period_by: Literal["Johannes", "Fabian"], repo_dir: str) -> None:
    """
    df: processed_DLC_data (corpus, piece, period, period_Johannes)
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="piece_distribution", repo_dir=repo_dir)

    if period_by == "Fabian":
        pieces_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first")
        )
    else:
        pieces_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Johannes=("period_Johannes", "first"),
        )

    pieces_df["corpus_id"] = pd.factorize(pieces_df["corpus"])[0] + 1
    pieces_df["piece_id"] = pd.factorize(pieces_df["piece"])[0] + 1
    pieces_df["period_id"] = pieces_df.apply(lambda row: determine_period_id(row=row, method=period_by), axis=1)

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

    if period_by == "Johannes":
        # plotting ________________________
        for b in [1650, 1750, 1800]:
            h.axvline(b, c="gray", ls="--")

        # save the fig
        # p = h.get_figure()
        plt.tight_layout()
        plt.savefig(f"{result_dir}fig_histogram_JP.pdf", dpi=200)

        # stats __________________________
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
        stats_df.to_latex(buf=f'{result_dir}corpus_stats_latex_table_JP.txt')
        del h

    elif period_by == "Fabian":

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
        piece_num = pieces_df.groupby(['corpus', 'period_Fabian']).agg(
            Piece_Number=('piece', 'count'),
            corpus_id=('corpus_id', "first"),
            period_id=('period_id', 'first')
        ).reset_index().sort_values(by=["corpus_id", "period_id"])

        stats_df = pd.merge(piece_num, metainfo_df, on='corpus', how='left')
        stats_df = stats_df[["period_Fabian", "Composer", "corpus", "Piece_Number"]]
        stats_df = stats_df.rename(
            columns={"period_Fabian": "Period", "corpus": "Corpus", "Piece_Number": "Piece Number"})
        stats_df['Corpus'] = stats_df['Corpus'].str.replace('_', ' ')
        stats_df.to_pickle(f'{result_dir}corpus_stats_table_FP.pickle')
        stats_df.to_latex(buf=f'{result_dir}corpus_stats_table_FP.txt')
        del h

    else:
        raise ValueError


# %% Analysis: Correlation analyses - Piece-level (global indices)

def _global_indices_pair_correlation_stats(df: pd.DataFrame,
                                           indices_pair: Tuple[
                                               Literal["WLC"], Literal["OLC", "WLD"]],
                                           method: Literal["pearson", "spearman"]) -> Tuple[float, float]:
    """
    df: assuming we take the piece-level df
    """
    if indices_pair == ("WLC", "OLC") or ("WLC", "WLD"):
        idx1, idx2 = indices_pair
        r = pg.corr(df[idx1], df[idx2], method=method).round(3)["r"].values[0]
        p_val = pg.corr(df[idx1], df[idx2], method=method).round(3)["p-val"].values[0]
    else:
        raise NotImplementedError
    return r, p_val


def _global_indices_corr_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
                                 indices_pair: Tuple[
                                     Literal["WLC"], Literal["OLC", "WLD"]],
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
                                 Literal["WLC"], Literal["OLC", "WLD"]],
                             period_by: Optional[Literal["Johannes", "Fabian"]],
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
    # ## current criteria: OLC >10 OR WLC >10
    # major_ER = [("chopin_mazurkas", "BI162-3op63-3"), ("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca")]
    # major_LR = [("tchaikovsky_seasons", "op37a12"), ("dvorak_silhouettes", "op08n12"),
    #             ("dvorak_silhouettes", "op08n01")]
    #
    # minor_ER = [("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca")]
    # minor_LR = [("bartok_bagatelles", "op06n12")] # WLC>10
    # # end of outlier list -------------

    if outliers_to_exclude_major:
        major_df = exclude_piece_from_corpus(df=major_df, corpus_piece_tups=outliers_to_exclude_major)
    else:
        major_df = major_df
    if outliers_to_exclude_minor:
        minor_df = exclude_piece_from_corpus(df=minor_df, corpus_piece_tups=outliers_to_exclude_minor)
    else:
        minor_df = minor_df

    if period_by == "Johannes":
        result_dfs = []
        for p in Johannes_periods:
            major_df_p = get_period_df(df=major_df, method="Johannes", period=p)
            minor_df_p = get_period_df(df=minor_df, method="Johannes", period=p)

            stats = _global_indices_corr_by_mode(major_df=major_df_p, minor_df=minor_df_p, indices_pair=indices_pair,
                                                 corr_method=corr_method)
            stats["period"] = p
            result_dfs.append(stats)
        result_df = pd.concat(result_dfs)
        file_anno = "_JP"

    elif period_by == "Fabian":
        result_dfs = []
        for p in Fabian_periods:
            major_df_p = get_period_df(df=major_df, method="Fabian", period=p)
            minor_df_p = get_period_df(df=minor_df, method="Fabian", period=p)

            stats = _global_indices_corr_by_mode(major_df=major_df_p, minor_df=minor_df_p, indices_pair=indices_pair,
                                                 corr_method=corr_method)
            stats["period"] = p
            result_dfs.append(stats)
        result_df = pd.concat(result_dfs)
        file_anno = "_FP"

    else:
        result_df = _global_indices_corr_by_mode(major_df=major_df, minor_df=minor_df, out_type="df",
                                                 indices_pair=indices_pair, corr_method=corr_method)
        file_anno = ""
    if save:
        idx1, idx2 = indices_pair
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        result_df.to_latex(buf=f'{sub_corr_analysis_folder}{idx1}_{idx2}_corr{file_anno}.txt')

    return result_df


def plot_piece_pairwise_indices_corr(df: pd.DataFrame,
                                     indices_pair: Tuple[
                                         Literal["WLC"], Literal["OLC", "WLD"]],
                                     period_by: Optional[Literal["Johannes", "Fabian"]],
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

    if period_by == "Johannes":
        fname = "JP"
        periods = Johannes_periods
        colors = ["#580F41", "#173573", "#c4840c", "#176373"]
        year_div = [f"<1650", f"1650-1750", f"1750-1800", f">1800"]
    elif period_by == "Fabian":
        fname = "FP"
        periods = Fabian_periods
        colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
        year_div = [f"<1662", f"1662-1763", f"1763-1821", f"1821-1869", f">1869"]
    else:
        fname = ''

    if period_by:
        # fig params:
        num_periods = len(periods)
        num_rows = 2
        num_cols = 4 if period_by == "Johannes" else 5
        fig, axs = plt.subplots(num_rows, num_cols,
                                layout="constrained",
                                figsize=(3 * num_periods + 1, 6))
        # period axes:
        for i, period in enumerate(periods):
            major_period_df = get_period_df(df=major_df, period=period, method=period_by)
            minor_period_df = get_period_df(df=minor_df, period=period, method=period_by)

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

    fig.supxlabel("\nWithin-Label Chromaticity (WLC)", fontsize=18, fontdict=dict(weight='bold'))
    if idx2 == "OLC":
        fig.supylabel("Out-of-Label Chromaticity (OLC)\n", fontsize=18, fontdict=dict(weight='bold'))
    elif idx2 == "WLD":
        fig.supylabel("Within-Label Dissonance (WLD)\n", fontsize=18, fontdict=dict(weight='bold'))
    else:
        raise ValueError

    # save plot
    if save:
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        plt.savefig(f'{sub_corr_analysis_folder}{idx1}_{idx2}_corr{fname}.pdf', dpi=200)


def plotly_piece_pairwise_indices_faceting(df: pd.DataFrame, indices_pair: Tuple[
    Literal["WLC"], Literal["OLC", "WLD"]], save: bool, repo_dir: str):
    """
        df: assuming we take the piece_level_indices_by_mode df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="correlation_analyses",
                                       repo_dir=repo_dir)

    idx1, idx2 = indices_pair
    fig = px.scatter(df, x=idx1, y=idx2, color="corpus", facet_col="period_Fabian", facet_row="localkey_mode",
                     hover_data="piece")
    if save:
        sub_corr_analysis_folder = f'{result_dir}global_indices/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)
        fig.write_html(f'{sub_corr_analysis_folder}plotly_{idx1}_{idx2}_scatter.html')


# %% Analysis: Correlation analyses - chord indices correlation in piece

def _chordlevel_indices_r_in_time(df: pd.DataFrame,
                                  mode: Literal["major", "minor"],
                                  indices_pair: Tuple[Literal["WLC", "OLC", "WLD"], Literal["WLC", "OLC", "WLD"]],
                                  repo_dir: str,
                                  save: bool = False):
    """
    df: assuming we take the chordlevel_indices_corr_by_piece df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="correlation_analyses", repo_dir=repo_dir)

    if indices_pair == ("WLC", "OLC"):
        col2look = "r_WLC_OLC"
    elif indices_pair == ("WLC", "WLD"):
        col2look = "r_WLC_WLD"
    else:
        raise ValueError

    # plots
    major_color = '#ee6c4d'
    minor_color = '#8db3c9'

    df.loc[:, "piece_year_jitter"] = rand_jitter(arr=df["piece_year"], scale=0.01)

    g = sns.jointplot(x="piece_year_jitter", y=col2look, data=df, hue="localkey_mode",
                      kind='scatter',
                      palette=[major_color, minor_color], alpha=0.7, legend='brief',
                      xlim=(1560, 1960), ylim=(-1, 1.1))

    g.set_axis_labels(xlabel="Year", ylabel="r", fontweight="bold", fontsize="large")

    # add a horizontal line at y=0:
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')

    g.fig.set_size_inches((8, 5))

    sns.move_legend(g.ax_joint, "lower left", title='Mode', frameon=True)

    # stats:

    pos_corr_num = (df[col2look] > 0).sum()
    neg_corr_num = (df[col2look] < 0).sum()
    not_corr_num = (df[col2look] == 0).sum()

    major_df = get_piece_df_by_localkey_mode(df=df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor")

    maj_pos_corr_num = (major_df[col2look] > 0).sum()
    maj_neg_corr_num = (major_df[col2look] < 0).sum()
    maj_not_corr_num = (major_df[col2look] == 0).sum()

    min_pos_corr_num = (minor_df[col2look] > 0).sum()
    min_neg_corr_num = (minor_df[col2look] < 0).sum()
    min_not_corr_num = (minor_df[col2look] == 0).sum()

    min_corr_row = df[df[col2look] == df[col2look].min()][["corpus", "piece", "localkey_mode", col2look]]
    max_corr_row = df[df[col2look] == 1][["corpus", "piece", "localkey_mode", col2look]]

    stats_dict = {
        f'{indices_pair} positive correlation num': pos_corr_num,
        f'{indices_pair} negative correlation num': neg_corr_num,
        f'{indices_pair} no correlation num': not_corr_num,
        f'{indices_pair} major pos correlation num': maj_pos_corr_num,
        f'{indices_pair} major neg correlation num': maj_neg_corr_num,
        f'{indices_pair} major not correlation num': maj_not_corr_num,
        f'{indices_pair} minor pos correlation num': min_pos_corr_num,
        f'{indices_pair} minor neg correlation num': min_neg_corr_num,
        f'{indices_pair} minor not correlation num': min_not_corr_num,

        f'min correlation corpus-piece-mode': min_corr_row,
        f'max correlation corpus-piece-mode': max_corr_row,

    }
    plt.show()

    if save:
        # save plots and data:
        sub_corr_analysis_folder = f'{result_dir}chord_indices_r_vals_in_pieces/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)

        plt.savefig(f'{sub_corr_analysis_folder}Fig_{col2look}_{mode}.pdf', dpi=200)
        with open(f'{sub_corr_analysis_folder}{col2look}_{mode}.txt', 'w') as f:
            for key, value in stats_dict.items():
                f.write('%s:%s\n\n' % (key, value))


def chordlevel_indices_r_vals(df: pd.DataFrame,
                              mode: Literal["major", "minor"],
                              indices_pair: Tuple[Literal["WLC", "OLC", "WLD"], Literal["WLC", "OLC", "WLD"]],
                              repo_dir: str,
                              save: bool = True):
    """
    df: assuming we take the chordlevel_indices_corr_by_piece df
    """
    pd.set_option('display.max_columns', None)

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="correlation_analyses", repo_dir=repo_dir)

    if indices_pair == ("WLC", "OLC"):
        col2look = "r_WLC_OLC"
    elif indices_pair == ("WLC", "WLD"):
        col2look = "r_WLC_WLD"
    else:
        raise ValueError

    # plots

    if mode == "major":
        # mode_color = '#ee6c4d'
        scatter_color = '#E18791'  # RED
        regline_color = '#be0e23'
    else:
        # mode_color = '#8db3c9'
        # scatter_color = '#C1B7CF' # PURPLE
        # regline_color = '#866aa3'
        scatter_color = '#83A0BE'  # BLUE
        regline_color = '#073E7F'

    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)

    # mode_df.loc[:, "piece_year_jitter"] = rand_jitter(arr=mode_df["piece_year"], scale=0.01)
    # mode_df["piece_year_jitter"] = rand_jitter(arr=mode_df["piece_year"], scale=0.01)

    g = sns.jointplot(x="piece_year", y=col2look, data=mode_df,
                      kind="reg", color=scatter_color, joint_kws={'x_jitter': 0.5,
                                                                  'scatter_kws': {'alpha': 0.5, 's': 25,
                                                                                  'linewidths': 0},
                                                                  'line_kws': {'linewidth': 2, 'color': regline_color
                                                                               }
                                                                  },
                      xlim=(1560, 1960), ylim=(-1, 1.1))

    g.set_axis_labels(xlabel="Year", ylabel="r", fontweight="bold", fontsize="large")

    # add a horizontal line at y=0:
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')

    g.fig.set_size_inches((6, 5))

    # sns.move_legend(g.ax_joint, "lower left", title='Mode', frameon=True)

    # stats:

    pos_corr_num = (mode_df[col2look] > 0).sum()
    neg_corr_num = (mode_df[col2look] < 0).sum()
    not_corr_num = (mode_df[col2look] == 0).sum()

    min_corr_row = mode_df[mode_df[col2look] == mode_df[col2look].min()][["corpus", "piece", "localkey_mode", col2look]]
    max_corr_row = mode_df[mode_df[col2look] == 1][["corpus", "piece", "localkey_mode", col2look]]

    slope, intercept, r_value, p_value, std_err = stats.linregress(mode_df.piece_year, mode_df[col2look])

    stats_dict = {
        f'{indices_pair} {mode} positive correlation num': pos_corr_num,
        f'{indices_pair} {mode} negative correlation num': neg_corr_num,
        f'{indices_pair} {mode} no correlation num': not_corr_num,

        f'min correlation corpus-piece-mode': min_corr_row,
        f'max correlation corpus-piece-mode': max_corr_row,

        f'regression stats:': '',
        f'slope': slope,
        f'intercept': intercept,
        f'r_value': r_value,
        f'p_value': p_value,
        f'std_err': std_err

    }

    if save:
        # save plots and data:
        sub_corr_analysis_folder = f'{result_dir}chord_indices_r_vals_in_pieces/'
        if not os.path.exists(sub_corr_analysis_folder):
            os.makedirs(sub_corr_analysis_folder)

        plt.savefig(f'{sub_corr_analysis_folder}Fig_{col2look}_{mode}.pdf', dpi=200)

        with open(f'{sub_corr_analysis_folder}{col2look}_{mode}.txt', 'w') as f:
            for key, value in stats_dict.items():
                f.write('%s:%s\n\n' % (key, value))


# %% Analysis: Mozart k331 theme and variations

def mozart_analysis(df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    """
    df: assuming we take the "chord_level_indices" df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="mozart_example", repo_dir=repo_dir)

    mozart = df[(df['corpus'] == 'mozart_piano_sonatas') & (df['piece'] == 'K331-1')]

    mozart["quarterbeats"] = mozart["quarterbeats"].apply(lambda x: fractions.Fraction(x))

    thema = mozart[(mozart["quarterbeats"].between(0, 45 / 2))]

    var1 = mozart[(mozart["quarterbeats"].between(111 / 2, 153 / 2))]

    var2 = mozart[(mozart["quarterbeats"].between(109, 261 / 2))]

    var3 = mozart[(mozart["quarterbeats"].between(162, 369 / 2))]

    var4 = mozart[(mozart["quarterbeats"].between(216, 477 / 2))]

    var5 = mozart[(mozart["quarterbeats"].between(1085 / 4, 293))]

    var6 = mozart[(mozart["quarterbeats"].between(325, 354))]

    dfs = []
    versions = ["Theme", "Var 1", "Var 2", "Var 3", "Var 4", "vVr 5", "Var 6"]
    for i, x in enumerate([thema, var1, var2, var3, var4, var5, var6]):
        x["version"] = versions[i]
        data = [x["version"].unique()[0], x["WLC"].mean(), x["OLC"].mean(), x["WLD"].mean()]
        cols = ["version", "WLC", "OLC", "WLD"]
        result = pd.DataFrame([data], columns=cols)
        dfs.append(result)

    results = pd.concat(dfs)

    results.to_latex(f'{result_dir}mozart_CI_table.txt', float_format="%.3f", index=False)


# %% Analysis: stats by corpus


def barplot_chromaticity_by_corpus(df: pd.DataFrame, repo_dir: str):
    """
    df: assume taking the corpora level indices df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 10), layout="constrained", sharey=True)

    df["corpus_pp"] = df["corpus"].map(corpus_prettyprint_dict)

    bar_width = 0.6
    major_color = '#95b5d3'
    minor_color = '#4b698f'

    sns.barplot(ax=axs[0], data=df, x="WLC", y="corpus_pp", hue="localkey_mode", errorbar=None, legend='brief',
                palette=[major_color, minor_color], width=bar_width)
    sns.barplot(ax=axs[1], data=df, x="OLC", y="corpus_pp", hue="localkey_mode", errorbar=None, legend=False,
                palette=[major_color, minor_color], width=bar_width)
    sns.barplot(ax=axs[2], data=df, x="WLD", y="corpus_pp", hue="localkey_mode", errorbar=None, legend=False,
                palette=[major_color, minor_color], width=bar_width)

    axs[0].set_ylabel("")
    axs[0].legend(loc="upper right", bbox_to_anchor=(-0.25, 1.02))
    axs[0].set_xlabel("WLC", fontweight="bold", fontsize=15)
    axs[1].set_xlabel("OLC", fontweight="bold", fontsize=15)
    axs[2].set_xlabel("WLD", fontweight="bold", fontsize=15)

    axs[2].set_xlim(left=0.2)

    for x in range(3):
        axs[x].spines['top'].set_visible(False)
        axs[x].spines['right'].set_visible(False)
        axs[x].spines['bottom'].set_visible(False)
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

    major_longform = pd.melt(major_df, id_vars=["corpus_pp", "corpus_id"], value_vars=["WLC", "OLC"],
                             var_name="index_type", value_name="value")
    minor_longform = pd.melt(minor_df, id_vars=["corpus_pp", "corpus_id"], value_vars=["WLC", "OLC"],
                             var_name="index_type", value_name="value")

    sns.barplot(ax=axs[0], data=major_longform, x="value", y="corpus_pp", hue="index_type", errorbar=None,
                legend='brief',
                palette=palette, width=bar_width)
    sns.barplot(ax=axs[1], data=minor_longform, x="value", y="corpus_pp", hue="index_type", errorbar=None, legend=False,
                palette=palette, width=bar_width)
    # sns.barplot(ax=axs[2], data=longform_df, x="WLD", y="corpus_pp", hue="localkey_mode", errorbar=None, legend=False,
    #             palette=[major_color, minor_color], width=bar_width)

    axs[0].set_ylabel("")
    axs[0].legend(loc="upper right", bbox_to_anchor=(-0.25, 1.02))
    # axs[0].set_xlabel("Major", fontweight="bold", fontsize=15)
    axs[0].set_title("Major", fontweight="bold", fontsize=15)
    axs[1].set_title("Minor", fontweight="bold", fontsize=15)
    # axs[1].set_xlabel("Minor", fontweight="bold", fontsize=15)
    # axs[2].set_xlabel("WLD", fontweight="bold", fontsize=15)
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


def barplot_dissonance_by_corpus(df: pd.DataFrame, repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="corpora_analyses", repo_dir=repo_dir)

    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")

    df["corpus_pp"] = df["corpus"].map(corpus_prettyprint_dict)
    bar_width = 0.6

    major_color = '#95b5d3'
    minor_color = '#4b698f'
    palette = [major_color, minor_color]
    sns.barplot(ax=ax, data=df, x="corpus_pp", y="WLD", hue="localkey_mode", errorbar=None, legend='brief',
                palette=palette, width=bar_width)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylim(0, 6)
    ax.set_ylabel("WLD", fontweight="bold", fontsize=13)
    ax.set_xlabel("Corpus", fontweight="bold", fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(frameon=False, ncol=2)

    # save fig
    fig_path = f'{result_dir}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(f'{fig_path}barplot_corpora_dissonance.pdf', dpi=200)


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

    print(f'Analysis: Chromaticity distribution (assumption of GPR) ...')
    plot_chrom_distribution(df=chromaticity_piece_by_mode, repo_dir=repo)

    print(f'Analysis: Basic stats for chromaticity and dissonance ...')
    stats_by_piece(df=chord_level_indices, repo_dir=repo)
    stats_by_corpus(df=chord_level_indices, repo_dir=repo)

    # Analysis: Piece distributions fig and table _________________________________________________________
    print(f'Analysis: Piece distributions fig and table ...')

    piece_distribution(df=prep_DLC, period_by="Fabian", repo_dir=repo)

    # Correlation Analysis: piece-level WLC and OLC ____________________________________
    print(f'Correlation Analysis: piece-level chromaticities ...')

    # manually remove outliers -------:

    ## current criteria: OLC >10 OR WLC >10
    WLC_OLC_major_outliers = [("chopin_mazurkas", "BI162-3op63-3"),
                              ("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca"),
                              ("tchaikovsky_seasons", "op37a12"), ("dvorak_silhouettes", "op08n12"),
                              ("dvorak_silhouettes", "op08n01")]

    WLC_OLC_minor_outliers = [("liszt_pelerinage", "161.04_Sonetto_47_del_Petrarca"),
                              ("bartok_bagatelles", "op06n12")]  # bartok_bagatelles: WLC>10
    # end of outlier list -------------

    for p in ["Fabian", None]:
        compute_piece_corr_stats(df=piece_level_indices_by_mode, indices_pair=("WLC", "OLC"), period_by=p,
                                 repo_dir=repo, corr_method="pearson",
                                 outliers_to_exclude_major=WLC_OLC_major_outliers,
                                 outliers_to_exclude_minor=WLC_OLC_minor_outliers,
                                 save=True)

        plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("WLC", "OLC"), period_by=p,
                                         repo_dir=repo, corr_method="pearson",
                                         outliers_to_exclude_major=WLC_OLC_major_outliers,
                                         outliers_to_exclude_minor=WLC_OLC_minor_outliers,
                                         save=True)

    plotly_piece_pairwise_indices_faceting(df=piece_level_indices_by_mode,
                                           indices_pair=("WLC", "OLC"), repo_dir=repo, save=True)

    # Correlation Analysis: piece-level WLC and WLD ____________________________________
    print(f'Correlation Analysis: piece-level chromaticity and dissonance ...')

    # manually remove outliers -------:
    WLC_WLD_major_outliers = None
    WLC_WLD_minor_outliers = [("bartok_bagatelles", "op06n12")]  # bartok_bagatelles: WLC>10

    for p in ["Fabian", None]:
        compute_piece_corr_stats(df=piece_level_indices_by_mode, indices_pair=("WLC", "WLD"), period_by=p,
                                 repo_dir=repo, corr_method="pearson",
                                 outliers_to_exclude_major=WLC_WLD_major_outliers,
                                 outliers_to_exclude_minor=WLC_WLD_minor_outliers,
                                 save=True)

        plot_piece_pairwise_indices_corr(df=piece_level_indices_by_mode, indices_pair=("WLC", "WLD"), period_by=p,
                                         repo_dir=repo, corr_method="pearson",
                                         outliers_to_exclude_major=WLC_WLD_major_outliers,
                                         outliers_to_exclude_minor=WLC_WLD_minor_outliers,
                                         save=True)

    plotly_piece_pairwise_indices_faceting(df=piece_level_indices_by_mode,
                                           indices_pair=("WLC", "WLD"), repo_dir=repo, save=True)

    # Correlation Analysis: chord-indices r-vals in pieces _____________________________
    print(f'Anallysis: Chord-level indices correlations (r-vals) in pieces...')

    # major mode:
    for m in ["major", "minor"]:
        chordlevel_indices_r_vals(df=chord_indices_r_vals_by_piece, mode=m, indices_pair=("WLC", "OLC"),
                                  repo_dir=repo, save=True)
        chordlevel_indices_r_vals(df=chord_indices_r_vals_by_piece, mode=m, indices_pair=("WLC", "WLD"),
                                  repo_dir=repo, save=True)

    # Analysis: Corpora-level indices:

    # barplot_dissonance_by_corpus(df=corpora_indices_by_mode, repo_dir=repo)
    barplot_chromaticity_by_corpus(df=corpora_indices_by_mode, repo_dir=repo)

    print(f'Fini!')


if __name__ == "__main__":
    full_analyses_set_for_paper()

    # dlc = load_file_as_df("/Users/xguan/Codes/chromaticism-codes/Data/prep_data/processed_DLC_data.pickle")
    #
    # plot_chord_size_across_time(df=dlc)
