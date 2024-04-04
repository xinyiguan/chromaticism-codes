import ast
import os
from typing import Literal

import numpy as np
import pandas as pd

from Code.utils.auxiliary import determine_period_Johannes, determine_period

pd.options.mode.chained_assignment = None  # default='warn'

from pitchtypes import SpelledPitchClass

from Code.utils.htypes import Numeral, Key
from Code.metrics import cumulative_distance_to_diatonic_set, pcs_dissonance_rank, \
    tpcs_to_ics
from Code.utils.util import flatten, flatten_to_list, safe_literal_eval, save_df, load_file_as_df


# Define function: Loading with additional preprocessing steps and saving

def process_DLC_data(data_path: str,
                     save: bool = True) -> pd.DataFrame:
    """
    An intermediate step for loading and process the filtered dcml harmonies tsv before computing chromaticity
    :param preprocessed_tsv:
    :param meatadata_tsv:
    :param save: whether to save the df for checking
    :return:
    """
    df = load_file_as_df(path=data_path)

    def find_localkey_spc(row):
        """get the local key tonic (in roman numeral) in spelled pitch class"""
        return Numeral.from_string(s=row["localkey"], k=Key.from_string(s=row["globalkey"])).key_if_tonicized().tonic

    def determine_mode(row):
        return "minor" if row["localkey"].islower() else "major"

    def localkey2C_dist(row):
        """get the fifths distance from the local key tonic to C"""
        return row["localkey_spc"].interval_to(SpelledPitchClass("C")).fifths()

    def correct_tpc_ref_center(row):
        return [x + row["localkey2C"] for x in row["tones_in_span_in_C"]]

    def tones_not_in_label(row):
        return [item for item in row['tones_in_span_in_lk'] if item not in row['within_label']]

    def tpc2spc(row):
        return [Key.get_spc_from_fifths(k=Key(tonic=row["localkey_spc"], mode=row["localkey_mode"]),
                                        fifth_step=t) for t in row["chord_tones"]]

    file_type = data_path.split(".")[-1]

    df["localkey_spc"] = df.apply(find_localkey_spc, axis=1)
    df["localkey_mode"] = df.apply(determine_mode, axis=1)
    df["localkey2C"] = df.apply(localkey2C_dist, axis=1)
    df["chord_tones_spc"] = df.apply(tpc2spc, axis=1)

    if file_type == "tsv":
        df["tones_in_span_in_C"] = df["all_tones_tpc_in_C"].apply(lambda s: list(ast.literal_eval(s))).apply(
            lambda lst: list(flatten(lst))).apply(lambda l: list(set(l)))

        df["added_tones"] = df["added_tones"].apply(lambda s: safe_literal_eval(s)).apply(flatten_to_list)
        df["chord_tones"] = df["chord_tones"].apply(lambda s: list(ast.literal_eval(s)))


    else:
        df["tones_in_span_in_C"] = df["all_tones_tpc_in_C"].apply(lambda lst: list(flatten(lst))).apply(
            lambda l: list(set(l)))

        df["added_tones"] = df["added_tones"].apply(flatten_to_list)
        df["chord_tones"] = df["chord_tones"].apply(flatten_to_list)

    df["tones_in_span_in_lk"] = df.apply(correct_tpc_ref_center, axis=1)
    df["within_label"] = df.apply(lambda row: [x for x in row["chord_tones"] + row["added_tones"]], axis=1)
    df["out_of_label"] = df.apply(tones_not_in_label, axis=1)

    df = df.assign(corpus_year=df.groupby("corpus")["piece_year"].transform("mean")).sort_values(
        ['corpus_year', 'piece_year']).reset_index(drop=True)

    df["period_Johannes"] = df.apply(determine_period_Johannes, axis=1)
    df["period"] = df.apply(determine_period, axis=1)

    try:
        df = df.drop(columns=['Unnamed: 0', 'all_tones_tpc_in_C'])
    except KeyError:
        pass

    if save:
        save_df(df=df, file_type="both", fname="processed_DLC_data", directory="../Data/prep_data/")

    return df


# Compute metrics : Chromaticity

def compute_chord_chromaticity(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    the input df should be the processed df with the load_dcml_harmonies_tsv()
    """

    # within-label chromaticity
    df["WLC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["within_label"],
                                                        diatonic_mode=row["localkey_mode"]), axis=1)
    # out-of-label chromaticity
    df["OLC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["out_of_label"],
                                                        diatonic_mode=row["localkey_mode"]), axis=1)

    result = df[
        ["corpus", "piece", "corpus_year", "piece_year", "period_Johannes", "period", "globalkey", "localkey", "localkey_spc", "localkey_mode",
         "quarterbeats", "chord", "tones_in_span_in_C", "tones_in_span_in_lk", "within_label", "WLC", "out_of_label", "OLC"]]
    if save:
        dir = f"../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result, file_type="both", directory=dir, fname="chromaticity_chord")

    return result


def compute_piece_chromaticity(df: pd.DataFrame, by: Literal["key_segment", "mode"],
                               save: bool = True) -> pd.DataFrame:
    def calculate_max_min_pc(x):
        if len(x) > 0:
            return max(x), min(x)
        else:  # hacking the zero-length all_tones
            return 0, 0

    df["max_WL"], df["min_WL"] = zip(*df["within_label"].apply(calculate_max_min_pc))
    df["max_OL"], df["min_OL"] = zip(*df["out_of_label"].apply(calculate_max_min_pc))

    if by == "key_segment":
        fname = f'chromaticity_piece_by_localkey'
        adj_check = (df.localkey != df.localkey.shift()).cumsum()
        result_df = df.groupby(['corpus', 'piece', adj_check], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period=("period", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),
            localkey_mode=("localkey_mode", "first"),

            max_WL=("max_WL", "max"),
            min_WL=("min_WL", "min"),
            max_OL=("max_OL", "max"),
            min_OL=("min_OL", "min"),

            WLC=("WLC", "mean"),
            OLC=("OLC", "mean")
        )

    elif by == "mode":
        fname=f'chromaticity_piece_by_mode'

        result_df = df.groupby(['corpus', 'piece', 'localkey_mode'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period=("period", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey_mode=("localkey_mode", "first"),

            max_WL=("max_WL", "max"),
            min_WL=("min_WL", "min"),
            max_OL=("max_OL", "max"),
            min_OL=("min_OL", "min"),

            WLC=("WLC", "mean"),
            OLC=("OLC", "mean")
        )
    else:
        raise ValueError()

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)
    result_df["WL_5th_range"] = (result_df["max_WL"] - result_df["min_WL"]).abs()
    result_df["OL_5th_range"] = (result_df["max_OL"] - result_df["min_OL"]).abs()

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        save_df(df=result_df, file_type="both", directory=dir, fname=fname)

    return result_df


# Compute metrics : Dissonance
def compute_chord_dissonance(df: pd.DataFrame,
                             save: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------

    df : pd.DataFrame
        should be the processed_DLC_data
    """

    # WITHIN-LABEL DISSONANCE
    df["ICs"] = df.apply(lambda row: tpcs_to_ics(tpcs=row["within_label"]), axis=1)
    df["WLD"] = df.apply(lambda row: pcs_dissonance_rank(tpcs=row["within_label"]), axis=1)

    result = df[
        ["corpus", "piece", "corpus_year", "piece_year", "period_Johannes", "period", "globalkey", "localkey",
         "duration_qb_frac", "chord", "within_label", "ICs", "WLD"]]

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result, file_type="both", directory=dir, fname="dissonance_chord")
    return result


def compute_piece_dissonance(df: pd.DataFrame, weighted: bool, save: bool = True) -> pd.DataFrame:
    if weighted:

        # get the duration of the all chords in a piece
        piece_dur_sum_df = df.groupby(['corpus', 'piece'])["duration_qb_frac"].sum().to_frame()

        # merge the two dataframes on 'corpus' and 'piece'
        merged_df = pd.merge(df, piece_dur_sum_df, on=['corpus', 'piece'], how='left')
        # rename
        merged_df.rename(columns={'duration_qb_frac_y': 'piece_dur'}, inplace=True)
        # Define a lambda function to compute the dissonance weighted by chord duration:
        merged_df["weighted_WLD"] = merged_df.WLD * (merged_df.duration_qb_frac_x / merged_df.piece_dur)
        result_df = merged_df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(

            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),

            piece_weighted_WLD=("weighted_WLD", "sum")
        )

    else:
        piece_chord_num_df = df.groupby(['corpus', 'piece']).size().rename("chord_num").to_frame()
        # merge the two dataframes on 'corpus' and 'piece'
        merged_df = pd.merge(df, piece_chord_num_df, on=['corpus', 'piece'], how='left')

        result_df = merged_df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(

            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period=("period", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),

            chord_num=("chord_num", "first"),
            total_WLD=("WLD", "sum"),
        )
        result_df["piece_avg_WLD"] = result_df.total_WLD / result_df.chord_num

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if weighted:
            fname = f"dissonance_piece_weighted_by_duration"
        else:
            fname = f"dissonance_piece_average"
        save_df(df=result_df, file_type="both", directory=dir, fname=fname)
    return result_df


# Combine metric dataframes
def combine_chord_level_indices(chord_chromaticity: pd.DataFrame, chord_dissonance: pd.DataFrame,
                        save: bool = True) -> pd.DataFrame:
    common_cols = ['corpus', 'piece', "corpus_year", "piece_year", "period_Johannes", "period", "globalkey", "localkey",
                   "chord", "within_label"]

    chromaticity = chord_chromaticity[common_cols + ["out_of_label", "WLC", "OLC"]]
    dissonance = chord_dissonance[common_cols + ["ICs", "WLD"]]

    chromaticity.within_label = chromaticity.within_label.apply(tuple)
    chromaticity.out_of_label = chromaticity.out_of_label.apply(tuple)
    dissonance.within_label = dissonance.within_label.apply(tuple)
    dissonance.ICs = dissonance.ICs.apply(tuple)

    result_df = chromaticity.merge(dissonance,
                                   on=common_cols,
                                   how="outer")[common_cols + ["out_of_label", "ICs", "WLC", "OLC", "WLD"]]

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result_df, file_type="both", directory=dir, fname="chord_level_indices")

    return result_df


def piece_level_indices(piece_chromaticity: pd.DataFrame, piece_dissonance: pd.DataFrame, fname: str,
                        save: bool = True) -> pd.DataFrame:
    common_cols = ['corpus', 'piece', "corpus_year", "piece_year", "globalkey", "localkey"]

    chromaticity = piece_chromaticity[common_cols]
    raise NotImplemented


if __name__ == "__main__":
    # process_DLC_data(data_path=f"../Data/prep_data/DLC_data.pickle", save=True)

    prep_DLC_data = load_file_as_df(path=f"../Data/prep_data/processed_DLC_data.pickle")

    print(f'Computing chord chromaticity indices ...')
    chord_chrom = compute_chord_chromaticity(df=prep_DLC_data)

    print(f'Computing piece-level chromaticity ...')
    piece_chrom_by_localkey = compute_piece_chromaticity(df=chord_chrom, by="key_segment")
    piece_chrom_by_mode = compute_piece_chromaticity(df=chord_chrom, by="mode")

    print(f'Finished chromaticity!')

    print(f'Computing chord dissonance indices ...')
    chord_dissonance = compute_chord_dissonance(df=prep_DLC_data)

    print(f'Computing piece dissonance indices ...')
    piece_dissonance_weighted_dur = compute_piece_dissonance(df=chord_dissonance, weighted=True, save=True)
    piece_dissonance_avg = compute_piece_dissonance(df=chord_dissonance, weighted=False, save=True)

    print(f'Finished dissonance!')

    chord_chrom = load_file_as_df(path=f"../Data/prep_data/for_analysis/chromaticity_chord.pickle")
    chord_diss = load_file_as_df(path=f"../Data/prep_data/for_analysis/dissonance_chord.pickle")

    print(f'Combing dfs for chromaticity and dissonance metrics ...')
    combine_chord_level_indices(chord_chromaticity=chord_chrom, chord_dissonance=chord_diss)

    print(f'Fini!')
