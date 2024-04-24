import fractions
import os
from typing import Literal, Optional, Tuple
import pandas as pd
import seaborn as sns
import pingouin as pg
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Code.utils.util import load_file_as_df, corpus_composer_dict, corpus_collection_dict
from Code.utils.auxiliary import create_results_folder, determine_period_id, get_period_df, Johannes_periods, \
    Fabian_periods, pprint_p_text, color_palette5, color_palette4, determine_group


# %% Analysis: Basic chrom, diss stats

# cols = ["min(WLC)", "avg(WLC)", "max(WLC)",
#         "min(OLC)", "avg(OLC)", "max(OLC)",
#         "min(WLD)", "avg(WLD)", "max(WLD)"]


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

    pieces_df.to_latex(f'{result_dir}basic_stats_by_piece.txt')

    return pieces_df


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

    pieces_df.to_latex(f'{result_dir}basic_stats_by_corpus.txt')

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

    result = pd.Series({
        "num sub-corpora": num_subcorpora,
        "num pieces": num_pieces,
        "num chords": num_chords,
        "num unique chord": num_unique_chords
    })

    result.to_string(f'{result_dir}DLC_corpus_stats.txt')

    return result


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

    plt.savefig(f'{result_dir}distribution_before_after_log_transformation.pdf', dpi=200)


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

        # save the fig
        # p = h.get_figure()
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


# %% Analysis: Chromaticity-Dissonance: correlation between chord-level WLC and WLD

def corr_chord_level_WLC_WLD(df: pd.DataFrame,
                             period_by: Literal["Johannes", "Fabian"],
                             repo_dir: str):
    """
    df: the chord-level indices dataframe containing all chord-level WLC and WLD values
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="chrom_diss_corr",
                                       repo_dir=repo_dir)
    if period_by == "Johannes":
        periods = Johannes_periods
        colors = ["#580F41", "#173573", "#c4840c", "#176373"]
        year_div = [f"<1650", f"1650-1750", f"1750-1800", f">1800"]
        fig_name_suffix = "JP"
    else:
        periods = Fabian_periods
        colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
        year_div = [f"<1662", f"1662-1763", f"1763-1821", f"1821-1869", f">1869"]
        fig_name_suffix = "FP"

    # fig params:
    num_periods = len(periods)
    num_rows = 1
    num_cols = 4 if period_by == "Johannes" else 5
    fig, axs = plt.subplots(num_rows, num_cols,
                            layout="constrained",
                            figsize=(4 * num_periods + 1, 5))

    for i, period in enumerate(periods):
        period_df = get_period_df(df=df, method=period_by, period=period)

        # Set column subtitle
        axs[i].set_title(year_div[i], fontweight="bold", fontsize=18, family="sans-serif")
        g = sns.regplot(ax=axs[i], data=period_df, x="WLC", y="WLD", color=colors[i])
        g.set_xlabel(f"WLC", fontsize=15)
        g.set_ylabel(f"WLD", fontsize=15)

        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        r = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["r"].values[0]
        p = pg.corr(period_df["WLC"], period_df["WLD"], method="pearson").round(3)["p-val"].values[0]

        p_text = pprint_p_text(p)

        # adding the text
        x_limit = axs[i].get_xlim()
        y_limit = axs[i].get_ylim()
        x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
        y_pos_1 = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

        g.text(x_pos, y_pos_1, f'r = {r}, {p_text}', fontsize=13, fontstyle='italic', ha='right',
               va='top')

    fig.savefig(f"{result_dir}fig_WLC_WLD_corr_by_{fig_name_suffix}.pdf", dpi=100)
    fig.savefig(f"{result_dir}fig_WLC_WLD_corr_by_{fig_name_suffix}.jpg", dpi=100)


# %% Analysis: Chromaticity-correlation between WLC and OLC

def _piece_chromaticity_df_by_mode(df: pd.DataFrame, mode: Literal["major", "minor"]) -> pd.DataFrame:
    if mode == "major":
        result_df = df[df['localkey_mode'].isin(['major'])]
    else:
        result_df = df[df['localkey_mode'].isin(['minor'])]

    return result_df


def perform_two_sample_ttest_for_mode_segment(df: pd.DataFrame, repo_dir: str) -> pd.DataFrame:
    """
    perform a two-sample t-test for major/minor mode segments

    df: take chromaticity_piece_by_mode df
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="piece_chrom_corr",
                                       repo_dir=repo_dir)

    major_df = _piece_chromaticity_df_by_mode(df=df, mode="major")
    minor_df = _piece_chromaticity_df_by_mode(df=df, mode="minor")

    WLC_major = major_df.loc[:, "WLC"]
    WLC_minor = minor_df.loc[:, "WLC"]

    OLC_major = major_df.loc[:, "OLC"]
    OLC_minor = minor_df.loc[:, "OLC"]

    WLC_ttest_result = pg.ttest(x=WLC_major, y=WLC_minor).rename(index={'T-test': 'WLC'})
    OLC_ttest_result = pg.ttest(x=OLC_major, y=OLC_minor).rename(index={'T-test': 'OLC'})

    ttest_result = pd.concat([WLC_ttest_result, OLC_ttest_result])
    ttest_result.to_latex(f'{result_dir}mode_ttest_result.txt')
    return ttest_result


def _WLC_OLC_correlation_stats(df: pd.DataFrame) -> Tuple[float, float]:
    r = pg.corr(df["WLC"], df["OLC"], method="pearson").round(3)["r"].values[0]
    p_val = pg.corr(df["WLC"], df["OLC"], method="pearson").round(3)["p-val"].values[0]
    return r, p_val


def _piece_chrom_corr_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
                              out_type: Literal["df", "tuple"] = "df") -> pd.DataFrame | Tuple:
    major_r, major_p = _WLC_OLC_correlation_stats(df=major_df)
    minor_r, minor_p = _WLC_OLC_correlation_stats(df=minor_df)

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


def compute_piece_chromaticity_corr_stats(df: pd.DataFrame,
                                          period_by: Optional[Literal["Johannes", "Fabian"]],
                                          repo_dir: str) -> pd.DataFrame:
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="piece_chrom_corr",
                                       repo_dir=repo_dir)

    _major_df = _piece_chromaticity_df_by_mode(df=df, mode="major")
    _minor_df = _piece_chromaticity_df_by_mode(df=df, mode="minor")

    if period_by == "Johannes":
        result_dfs = []
        for p in Johannes_periods:
            major_df = get_period_df(df=_major_df, method="Johannes", period=p)
            minor_df = get_period_df(df=_minor_df, method="Johannes", period=p)

            stats = _piece_chrom_corr_by_mode(major_df=major_df, minor_df=minor_df)
            stats["period"] = p
            result_dfs.append(stats)
        result_df = pd.concat(result_dfs)
        result_df.to_latex(buf=f'{result_dir}piece_chrom_corr_JP.txt')

    elif period_by == "Fabian":
        result_dfs = []
        for p in Fabian_periods:
            major_df = get_period_df(df=_major_df, method="Fabian", period=p)
            minor_df = get_period_df(df=_minor_df, method="Fabian", period=p)

            stats = _piece_chrom_corr_by_mode(major_df=major_df, minor_df=minor_df)
            stats["period"] = p
            result_dfs.append(stats)
        result_df = pd.concat(result_dfs)
        result_df.to_latex(buf=f'{result_dir}piece_chrom_corr_FP.txt')

    else:
        result_df = _piece_chrom_corr_by_mode(major_df=_major_df, minor_df=_minor_df, out_type="df")
        result_df.to_latex(buf=f'{result_dir}piece_chrom_corr.txt')

    return result_df


def plot_piece_chromaticity_WLC_OLC_corr(df: pd.DataFrame, period_by: Literal["Johannes", "Fabian"], repo_dir: str):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results",
                                       analysis_name="piece_chrom_corr",
                                       repo_dir=repo_dir)

    major_df = _piece_chromaticity_df_by_mode(df=df, mode="major")
    minor_df = _piece_chromaticity_df_by_mode(df=df, mode="minor")

    if period_by == "Johannes":
        periods = Johannes_periods
        colors = ["#580F41", "#173573", "#c4840c", "#176373"]
        year_div = [f"<1650", f"1650-1750", f"1750-1800", f">1800"]
    else:
        periods = Fabian_periods
        colors = ["#6e243b", "#173573", "#c4840c", "#176373", "#816683"]
        year_div = [f"<1662", f"1662-1763", f"1763-1821", f"1821-1869", f">1869"]

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

        major_r, major_p, minor_r, minor_p = _piece_chrom_corr_by_mode(major_df=major_period_df,
                                                                       minor_df=minor_period_df,
                                                                       out_type="tuple")
        major_p_txt = pprint_p_text(major_p)
        minor_p_txt = pprint_p_text(minor_p)

        # plotting
        row_index = i // num_cols
        col_index = i % num_cols
        axs[row_index, col_index].set_title(year_div[i], fontweight="bold", fontsize=18, family="sans-serif")
        ma = sns.regplot(ax=axs[row_index, col_index], data=major_period_df, x="WLC", y="OLC",
                         color=colors[i], marker='o', scatter_kws={'s': 10})
        mi = sns.regplot(ax=axs[row_index + 1, col_index], data=minor_period_df, x="WLC", y="OLC",
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

        else:
            ma.set_xlabel("")
            mi.set_ylabel("")

    fig.supxlabel("Within-Label Chromaticity (WLC)", fontsize=15, fontdict=dict(weight='bold'))
    fig.supylabel("Out-of-Label Chromaticity (OLC)", fontsize=15, fontdict=dict(weight='bold'))
    # save plot
    plt.savefig(f'{result_dir}fig_chrom_corr_period{period_by}.pdf', dpi=200)
    plt.savefig(f'{result_dir}fig_chrom_corr_period{period_by}.jpg', dpi=200)


# %% Analysis: Source of chromaticity in a piece (WLC-OLC percentage)

def plor_scatter_chromaticity_source_ratio_across_time(df: pd.DataFrame,
                                                       mode: Literal["major", "minor"],
                                                       era_division: Literal["Fabian", "Johannes"],
                                                       repo_dir: str):
    """
    df: assuming we take the "chromaticity_piece_[mode]" df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="chromaticity_source", repo_dir=repo_dir)

    if era_division == "Fabian":
        color_palette = color_palette5
    else:
        color_palette = color_palette4

    era_div = f'period_{era_division}'

    s = sns.scatterplot(data=df, x="piece_year", y="WLC_percentage",
                        hue=era_div, palette=color_palette)
    s.axhline(0.5, c="gray", ls="--")

    plt.show()


def ridge_plot_chromaticity_ratio(df: pd.DataFrame,
                                  mode: Literal["major", "minor"],
                                  step_by: Literal["group", "period_Fabian", "period_Johannes"],
                                  year_interval: Optional[Literal[25, 50]],
                                  repo_dir: str):
    """
    df: assuming we take the "chromaticity_piece_[mode]" df
    """
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="chromaticity_source", repo_dir=repo_dir)
    df = df.sort_values(by=['piece_year'])

    if step_by == "group":
        assert year_interval is not None
        df["step_by"] = df.apply(lambda row: determine_group(row=row, interval=year_interval), axis=1)
    elif step_by == "period_Fabian":
        df["step_by"] = df["period_Fabian"]
    else:
        df["step_by"] = df["period_Johannes"]

    num_axes = len(df["step_by"].unique())

    # Initialize the FacetGrid object
    palette = sns.cubehelix_palette(num_axes, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="step_by", hue="step_by", aspect=6, height=.8, palette=palette)

    # Draw the densities
    bw_adjust = .5
    linewidth = 1
    g.map(sns.kdeplot, "WLC_percentage",
          bw_adjust=bw_adjust,
          # clip_on=False,
          fill=True, alpha=1, linewidth=linewidth)

    # g.map(sns.kdeplot, "WLC_percentage",
    #       # clip_on=False,
    #       color="w", lw=linewidth, bw_adjust=bw_adjust)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .4, label, color=color, fontsize="small",
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "step_by")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")

    g.set(yticks=[], ylabel="")
    g.set(xticks=[], xlabel="within-label chromaticity percentage")
    g.despine(bottom=True, left=True)

    # iterate over axes of FacetGrid
    for ax in g.axes.flat:
        labels = [0, 0.5, 1]
        ax.set_xticks(labels)

    g.fig.tight_layout(w_pad=0.25)
    g.fig.suptitle(f"{mode}")

    fig_name = f'ridge_plot_{step_by}_{mode}.pdf'

    plt.savefig(f'{result_dir}{fig_name}', dpi=200)


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
    versions = ["thema", "var1", "var2", "var3", "var4", "var5", "var6"]
    for i, x in enumerate([thema, var1, var2, var3, var4, var5, var6]):
        x["version"] = versions[i]
        data = [x["version"].unique()[0], x["WLC"].mean(), x["OLC"].mean()]
        cols = ["version", "WLC", "OLC"]
        result = pd.DataFrame([data], columns=cols)
        dfs.append(result)

    results = pd.concat(dfs)

    results.to_latex(f'{result_dir}mozart_CI_table.txt', index=False)



if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'

    chord_indices = load_file_as_df(
        "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/chord_level_indices.pickle")
    mozart_analysis(chord_indices, repo_dir=repo)

    print(f'DLC corpus basic stats ...')
    chord_level_df = load_file_as_df(path=f'{repo}Data/prep_data/processed_DLC_data.pickle')
    DLC_corpus_stats(chord_level_df=chord_level_df, repo_dir=repo)

    # Analysis: Basic stats _________________________________________________________
    print(f'Analysis: Basic stats for chromaticity and dissonance __________________')
    chrom_by_mode = load_file_as_df(path=f'{repo}Data/prep_data/for_analysis/chromaticity_piece_by_mode.pickle')
    plot_chrom_distribution(df=chrom_by_mode, repo_dir=repo)

    print(f'Analysis: Basic stats for chromaticity and dissonance __________________')
    chord_level_df = load_file_as_df(path=f'{repo}Data/prep_data/for_analysis/chord_level_indices.pickle')
    stats_by_piece(df=chord_level_df, repo_dir=repo)
    stats_by_corpus(df=chord_level_df, repo_dir=repo)

    # Analysis: Piece distributions fig and table _________________________________________________________
    print(f'Analysis: Piece distributions fig and table _____________________________________')
    prep_DLC_df = load_file_as_df(path=f"{repo}Data/prep_data/processed_DLC_data.pickle")

    piece_distribution(df=prep_DLC_df, period_by="Fabian", repo_dir=repo)
    print(f'Finished piece distribution analysis for period division (Fabian division)...')

    piece_distribution(df=prep_DLC_df, period_by="Johannes", repo_dir=repo)
    print(f'Finished piece distribution analysis for period division (Johannes division)...')

    # Analysis: Chromaticity-Dissonance corr: chord-level WLC and WLD ____________________________________

    print(f'Analysis: correlation between chord-level chromaticity and dissonance _______________________')
    chord_indices_df = load_file_as_df(path=f"{repo}Data/prep_data/for_analysis/chord_level_indices.pickle")
    print(f'Starting the corr analyses...')
    corr_chord_level_WLC_WLD(df=chord_indices_df, period_by="Johannes", repo_dir=repo)
    corr_chord_level_WLC_WLD(df=chord_indices_df, period_by="Fabian", repo_dir=repo)

    # Analysis: chromaticity correlation

    print(f'Analysis: correlation between WLC and OLC by periods ______________________')
    chromaticity_df = load_file_as_df(path=f"{repo}Data/prep_data/for_analysis/chromaticity_piece_by_mode.pickle")

    print(f'    t-test for major and minor segment groups ...')
    ttest_for_mode_separation = perform_two_sample_ttest_for_mode_segment(df=chromaticity_df, repo_dir=repo)

    print(f'    correlation between WLC and OLC ...')
    chrom_corr_by_JP = compute_piece_chromaticity_corr_stats(df=chromaticity_df, period_by="Johannes", repo_dir=repo)
    chrom_corr_by_FP = compute_piece_chromaticity_corr_stats(df=chromaticity_df, period_by="Fabian", repo_dir=repo)
    chrom_corr = compute_piece_chromaticity_corr_stats(df=chromaticity_df, period_by=None, repo_dir=repo)
    plot_piece_chromaticity_WLC_OLC_corr(df=chromaticity_df, period_by="Fabian", repo_dir=repo)
    plot_piece_chromaticity_WLC_OLC_corr(df=chromaticity_df, period_by="Johannes", repo_dir=repo)


    # Analysis: chromaticity source
    print(f'Analysis: ridge plot for sources of chromaticism ______________________')

    chromaticity_major = load_file_as_df(
        "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/chromaticity_piece_major.pickle")
    chromaticity_minor = load_file_as_df(
        "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/chromaticity_piece_minor.pickle")

    ridge_plot_chromaticity_ratio(df=chromaticity_major, mode="major", year_interval=None, step_by="period_Fabian",
                                  repo_dir=repo)
    ridge_plot_chromaticity_ratio(df=chromaticity_minor, mode="minor", year_interval=None, step_by="period_Fabian",
                                  repo_dir=repo)
    ridge_plot_chromaticity_ratio(df=chromaticity_major, mode="major", year_interval=None, step_by="period_Johannes",
                                  repo_dir=repo)
    ridge_plot_chromaticity_ratio(df=chromaticity_minor, mode="minor", year_interval=None, step_by="period_Johannes",
                                  repo_dir=repo)


    print(f'Fini!')
