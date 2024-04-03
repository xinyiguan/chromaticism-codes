import os
from typing import Literal

import pandas as pd

from Code.utils.util import load_file_as_df

# %% Analysis: piece distribution in periods
import os
from typing import Literal
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from Code.utils.util import load_file_as_df, corpus_composer_dict, corpus_collection_dict
from Code.utils.auxiliary import create_results_folder, determine_period_Johannes, determine_period, determine_period_id


def piece_distribution(df: pd.DataFrame, period_by: Literal["Johannes", "old"], repo_dir: str):
    """
    df: DLC_data (corpus, piece, period, period_Johannes)
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

        metainfo_df = pd.DataFrame.from_dict(corpus_composer_dict, orient='index').reset_index()
        metainfo_df["Collection"] = metainfo_df['index'].map(corpus_collection_dict)
        metainfo_df = metainfo_df.rename(columns={'index': 'corpus', 0: 'Composer'})
        stats_df = pd.merge(piece_num, metainfo_df, on='corpus', how='left')
        stats_df = stats_df[["period_Johannes", "Composer", "corpus", "Piece_Number"]]
        stats_df = stats_df.rename(
            columns={"period_Johannes": "Period", "corpus": "Corpus", "Piece_Number": "Piece Number"})
        stats_df.to_pickle(f'{result_dir}corpus_stats_table.pickle')
        stats_df.to_latex(buf=f'{result_dir}corpus_stats_table')

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

        t1 = round(t1)
        t2 = round(t2)
        t3 = round(t3)
        t4 = round(t4)

        # save the fig
        p = h.get_figure()
        p.savefig(f"{result_dir}fig_histogram_period.pdf", dpi=300)

        ## stats __________________________


    else:
        raise ValueError


# %% Analysis: correlation between chord-level WLC and WLD

def corr_chord_level_WLC_WLD(df: pd.DataFrame, by_period: bool = True, save_results: bool = True):
    """
    df: the chord-level indices dataframe containing all chord-level WLC and WLD values
    """
    df = load_file_as_df(path="../Data/prep_data/for_analysis/chord_level_indices.pickle")

    # period division conforming to Johannes' thesis


# %% Analysis:

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    repo_dir = '/Users/xguan/Codes/chromaticism-codes/'
    df = load_file_as_df(path=f"{repo_dir}/Data/prep_data/for_analysis/dissonance_piece_average.pickle")
    piece_distribution(df=df, period_by="Johannes", repo_dir=repo_dir)
