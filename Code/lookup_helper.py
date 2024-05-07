import json
import os
from typing import Tuple, Literal

import matplotlib.pyplot as plt
import pandas as pd

from Code.analysis import get_piece_df_by_localkey_mode
from Code.utils.auxiliary import create_results_folder
from Code.utils.util import load_file_as_df
import seaborn as sns

pd.set_option('display.max_columns', None)


def find_outlier(df: pd.DataFrame, year_range: Tuple[int, int],
                 index_type: Literal["WLC", "OLC", "avg_WLD"],
                 outlier_type: Literal["max", "min"]) -> pd.DataFrame:
    df_cut_time = df[(df['piece_year'] >= year_range[0]) & (df['piece_year'] <= year_range[1])]
    if outlier_type == "max":
        outlier_col_loc = df_cut_time[index_type].idxmax()
    else:
        outlier_col_loc = df_cut_time[index_type].idxmin()

    res = df_cut_time.loc[outlier_col_loc]

    print(res)


def find_corpora(df: pd.DataFrame, year_range: Tuple[int, int],
                 lookup_col: str) -> pd.DataFrame:
    df_cut_time = df[(df['piece_year'] >= year_range[0]) & (df['piece_year'] <= year_range[1])]

    res = df_cut_time[lookup_col].unique()

    print(res)


def count_chord_occurrence(df: pd.DataFrame, chord: str, repo_dir: str):
    """
    df: assume chord_level_indices
    """

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="chord_stats", repo_dir=repo_dir)

    def determine_mode(row):
        return "minor" if row["localkey"].islower() else "major"

    df["localkey_mode"] = df.apply(determine_mode, axis=1)

    major = df[chord_level['localkey_mode'].isin(['major'])]
    minor = df[chord_level['localkey_mode'].isin(['minor'])]

    num_in_major = major[major["chord"] == chord].shape[0]
    num_in_minor = minor[minor["chord"] == chord].shape[0]

    print(num_in_major)
    print(num_in_minor)

    percent_in_major = num_in_major / major.shape[0]
    percent_in_minor = num_in_minor / minor.shape[0]

    res = {
        'num in major': num_in_major,
        'percent in major': percent_in_major,
        'num in minor': num_in_minor,
        'percent in minor': percent_in_minor
    }
    #
    # with open(f'{result_dir}{chord}_occurrence_stats.txt', 'w') as file:
    #     file.write(json.dumps(res))  # use `json.loads` to do the reverse

    return (num_in_major, percent_in_major), (num_in_minor, percent_in_minor)


def count_piece_percentage_by_threshold(df: pd.DataFrame, mode: Literal["major", "minor"],
                                        index_type: Literal["WLC", "OLC", "WLD"],
                                        threshold: float):
    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)
    total_pieces = mode_df.shape[0]
    desired_piece_count = df[df[index_type] > threshold].shape[0]

    res = (total_pieces - desired_piece_count) / total_pieces
    return res


def get_piece_percentage_by_threshold(df: pd.DataFrame, threshold: float,
                                      repo_dir: str):
    major_df = get_piece_df_by_localkey_mode(df=df, mode='major')
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor")

    total_major_p = major_df.shape[0]
    WLC_major_threshold_p = major_df[major_df["WLC"] > threshold].shape[0]
    WLC_minor_threshold_p = minor_df[minor_df["WLC"] > threshold].shape[0]

    WLC_major_pctg = (total_major_p - WLC_major_threshold_p) / total_major_p
    WLC_minor_pctg = (total_major_p - WLC_minor_threshold_p) / total_major_p

    OLC_major_threshold_p = major_df[major_df["OLC"] > threshold].shape[0]
    OLC_minor_threshold_p = minor_df[minor_df["OLC"] > threshold].shape[0]

    OLC_major_pctg = (total_major_p - OLC_major_threshold_p) / total_major_p
    OLC_minor_pctg = (total_major_p - OLC_minor_threshold_p) / total_major_p

    WLD_major_threshold_p = major_df[major_df["WLD"] > threshold].shape[0]
    WLD_minor_threshold_p = minor_df[minor_df["WLD"] > threshold].shape[0]

    WLD_major_pctg = (total_major_p - WLD_major_threshold_p) / total_major_p
    WLD_minor_pctg = (total_major_p - WLD_minor_threshold_p) / total_major_p

    res_dict = {
        'threshold': threshold,
        'WLC major': WLC_major_pctg,
        'WLC minor': WLC_minor_pctg,
        'OLC major': OLC_major_pctg,
        'OLC minor': OLC_minor_pctg,
        'WLD major': WLD_major_pctg,
        'WLD minor': WLD_minor_pctg
    }

    print(res_dict)


def get_piece_subdf(df: pd.DataFrame, corpus: str, piece: str):
    res_df = df[(df["corpus"] == corpus) & (df["piece"] == piece)]
    return res_df




if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'

    chord_level = load_file_as_df(
        "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/chord_level_indices.pickle")

    # schumann=get_piece_subdf(df=chord_level, corpus="schumann_liederkreis", piece="op39n08")
    # sns.regplot(x="WLC", y="WLD", data=schumann)
    # plt.show()
    #
    # assert False
    # piece_indices = load_file_as_df(
    #     "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/piece_level_indices_by_mode.pickle")
    # get_piece_percentage_by_threshold(df=piece_indices, threshold=1, repo_dir=repo)
    #
    # assert False
    res = count_chord_occurrence(df=chord_level, chord="bII", repo_dir=repo)
    print(res)

    assert False
    minor = get_piece_df_by_localkey_mode(df=piece_indices, mode="minor")
    major = get_piece_df_by_localkey_mode(df=piece_indices, mode="major")
    find_corpora(df=major, year_range=(1850, 1880), lookup_col="corpus")

    diss_minor_1800_max = find_outlier(df=minor, year_range=(1780, 1830),
                                       index_type="avg_WLD", outlier_type="max")

    print(f'______________')

    diss_minor_1800_min = find_outlier(df=minor, year_range=(1780, 1830),
                                       index_type="avg_WLD", outlier_type="min")

    print(f'______________')

    diss_major_1650_max = find_outlier(df=major, year_range=(1650, 1670),
                                       index_type="avg_WLD", outlier_type="max")

    diss_minor_1650_max = find_outlier(df=major, year_range=(1650, 1670),
                                       index_type="avg_WLD", outlier_type="min")
