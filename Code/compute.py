import ast
import os
from typing import Optional, Callable, Literal

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from pitchtypes import SpelledPitchClass

from Code.utils.htypes import Numeral, Key
from Code.metrics import tone_to_diatonic_set_distance, cumulative_distance_to_diatonic_set, pcs_dissonance_rank, \
    tpcs_to_ics
from Code.utils.util import flatten, flatten_to_list, safe_literal_eval, save_df, load_file_as_df


# Define functions: Loading with additional preprocessing steps and saving
def process_DLC_data_v1(preprocessed_tsv: str,
                        save: bool = True) -> pd.DataFrame:
    """
    An intermediate step for loading and process the filtered dcml harmonies tsv before computing chromaticity
    :param preprocessed_tsv:
    :param meatadata_tsv:
    :param save: whether to save the df for checking
    :return:
    """
    df = pd.read_csv(preprocessed_tsv, sep="\t")

    def find_lk_spc(row):
        """get the local key tonic (in roman numeral) in spelled pitch class"""
        return Numeral.from_string(s=row["localkey"], k=Key.from_string(s=row["globalkey"])).key_if_tonicized().tonic

    def lk2c_dist(row):
        """get the fifths distance from the local key tonic to C"""
        return row["localkey_spc"].interval_to(SpelledPitchClass("C")).fifths()

    def correct_tpc_ref_center(row):
        return [x + row["lk2C"] for x in row["flatten_tones_in_span_in_C"]]

    def tones_not_in_ct_root(row):
        return [item for item in row['tones_in_span'] if item not in row['ct'] + [row['root']]]

    def determine_mode(row):
        return "minor" if row["localkey"].islower() else "major"

    df["lk_mode"] = df.apply(determine_mode, axis=1)

    df["localkey_spc"] = df.apply(find_lk_spc, axis=1)

    df["lk2C"] = df.apply(lk2c_dist, axis=1)

    # flatten and get unique tpc in the list in tones_in_span_in_C col
    df["flatten_tones_in_span_in_C"] = df["tones_in_span_in_C"].apply(lambda s: list(ast.literal_eval(s))).apply(
        lambda lst: list(flatten(lst))).apply(lambda l: list(set(l)))

    # correct the tpc to reference to local key tonic
    df["tones_in_span"] = df.apply(correct_tpc_ref_center, axis=1)

    df["added_tones"] = df["added_tones"].apply(lambda s: safe_literal_eval(s)).apply(flatten_to_list)
    df["chord_tones"] = df["chord_tones"].apply(lambda s: list(ast.literal_eval(s)))

    df["ct"] = df.apply(lambda row: [x for x in row["chord_tones"] + row["added_tones"] if x != row["root"]], axis=1)
    df["nct"] = df.apply(tones_not_in_ct_root, axis=1)

    df = df.assign(corpus_year=df.groupby("corpus")["piece_year"].transform(np.mean)).sort_values(
        ['corpus_year', 'piece_year']).reset_index(drop=True)
    df = df.drop(columns=['Unnamed: 0'])

    if save:
        save_df(df=df, fname="processed_DLC_data", directory="../Data/prep_data/")

    return df


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

    def localkey2C_dist(row):
        """get the fifths distance from the local key tonic to C"""
        return row["localkey_spc"].interval_to(SpelledPitchClass("C")).fifths()

    def correct_tpc_ref_center(row):
        return [x + row["localkey2C"] for x in row["tones_in_span_in_C"]]

    def tones_not_in_label(row):
        return [item for item in row['tones_in_span_in_lk'] if item not in row['within_label']]

    file_type = data_path.split(".")[-1]

    df["localkey_spc"] = df.apply(find_localkey_spc, axis=1)
    df["localkey2C"] = df.apply(localkey2C_dist, axis=1)

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

    try:
        df = df.drop(columns=['Unnamed: 0', 'all_tones_tpc_in_C'])
    except KeyError:
        pass

    if save:
        save_df(df=df, file_type="both", fname="processed_DLC_data", directory="../Data/prep_data/")

    return df


# Define functions: Chromaticity
def compute_chord_chromaticity_v1(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    the input df should be the processed df with the load_dcml_harmonies_tsv()
    """

    # the distance of the root to the closet members of the diatonic set generated by the localkey tonic.
    df["RC"] = df.apply(lambda row: tone_to_diatonic_set_distance(tone=int(row["root"]),
                                                                  tonic=None,
                                                                  diatonic_mode=row["lk_mode"]), axis=1)

    # the cumulative distance of the chord tones to the local key scale set
    df["CTC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["ct"], diatonic_mode=row["lk_mode"]), axis=1)

    # the cumulative distance of the non-chord tones to the local key scale set
    df["NCTC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["nct"], diatonic_mode=row["lk_mode"]),
        axis=1)

    result = df[
        ["corpus", "piece", "corpus_year", "piece_year", "globalkey", "localkey", "localkey_spc", "quarterbeats",
         "tones_in_span_in_C", "tones_in_span", "chord", "root", "RC", "ct", "CTC", "nct", "NCTC"]]

    if save:
        save_df(df=result, file_type="both", directory="../Data/prep_data/", fname="chromaticity_chord")

    return result


def compute_piece_chromaticity_v1(df: pd.DataFrame, save: bool = True, compute_full: bool = False) -> pd.DataFrame:
    def calculate_max_min_pc(x):
        if len(x) > 0:
            return max(x), min(x)
        else:  # hacking the zero-length all_tones
            return 0, 0

    df["max_ct"] = df["ct"].apply(lambda x: max(x))
    df["min_ct"] = df["ct"].apply(lambda x: min(x))

    df["max_nct"], df["min_nct"] = zip(*df["nct"].apply(calculate_max_min_pc))

    if compute_full:
        result_df = df.groupby(['corpus', 'piece'], as_index=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),

            max_root=("root", "max"),
            min_root=("root", "min"),

            max_ct=("max_ct", "max"),
            min_ct=("min_ct", "min"),

            max_nct=("max_nct", "max"),
            min_nct=("min_nct", "min"),

            mean_r_chromaticity=("RC", "mean"),
            max_r_chromaticity=("RC", "max"),
            min_r_chromaticity=("RC", "min"),

            mean_ct_chromaticity=("CTC", "mean"),
            max_ct_chromaticity=("CTC", "max"),
            min_ct_chromaticity=("CTC", "min"),

            mean_nct_chromaticity=("NCTC", "mean"),
            max_nct_chromaticity=("NCTC", "max"),
            min_nct_chromaticity=("NCTC", "min")
        )

    else:
        result_df = df.groupby(['corpus', 'piece'], as_index=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),

            max_root=("root", "max"),
            min_root=("root", "min"),

            max_ct=("max_ct", "max"),
            min_ct=("min_ct", "min"),

            max_nct=("max_nct", "max"),
            min_nct=("min_nct", "min"),

            tones_in_span=("tones_in_span", "first"),
            tones_in_span_in_C=("tones_in_span_in_C", "first"),
            root=("root", "first"),
            RC=("RC", "mean"),
            ct=("ct", "first"),
            CTC=("CTC", "mean"),
            nct=("nct", "first"),
            NCTC=("NCTC", "mean")
        )

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)
    result_df["r_fifths_range"] = (result_df["max_root"] - result_df["min_root"]).abs()
    result_df["ct_fifths_range"] = (result_df["max_ct"] - result_df["min_ct"]).abs()
    result_df["nct_fifths_range"] = (result_df["max_nct"] - result_df["min_nct"]).abs()

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = list(range(1, len(result_df) + 1))

    if save:
        save_df(df=result_df, file_type="both", directory="../Data/prep_data/", fname="chromaticity_piece")

    return result_df


def compute_chord_chromaticity(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    the input df should be the processed df with the load_dcml_harmonies_tsv()
    """

    def determine_mode(row):
        return "minor" if row["localkey"].islower() else "major"

    df["localkey_mode"] = df.apply(determine_mode, axis=1)

    # within-label chromaticity
    df["WLC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["within_label"],
                                                        diatonic_mode=row["localkey_mode"]), axis=1)
    # out-of-label chromaticity
    df["OLC"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["out_of_label"],
                                                        diatonic_mode=row["localkey_mode"]), axis=1)

    result = df[
        ["corpus", "piece", "corpus_year", "piece_year", "globalkey", "localkey", "localkey_spc", "localkey_mode",
         "quarterbeats",
         "tones_in_span_in_C", "tones_in_span_in_lk", "chord", "within_label", "WLC", "out_of_label", "OLC"]]
    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result, file_type="both", directory=dir, fname="chromaticity_chord")

    return result


def compute_piece_chromaticity_by_key_segment(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    def calculate_max_min_pc(x):
        if len(x) > 0:
            return max(x), min(x)
        else:  # hacking the zero-length all_tones
            return 0, 0

    df["max_WL"], df["min_WL"] = zip(*df["within_label"].apply(calculate_max_min_pc))
    df["max_OL"], df["min_OL"] = zip(*df["out_of_label"].apply(calculate_max_min_pc))

    adj_check = (df.localkey != df.localkey.shift()).cumsum()

    result_df = df.groupby(['corpus', 'piece', adj_check], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
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

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)
    result_df["WL_5th_range"] = (result_df["max_WL"] - result_df["min_WL"]).abs()
    result_df["OL_5th_range"] = (result_df["max_OL"] - result_df["min_OL"]).abs()

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        save_df(df=result_df, file_type="both", directory=dir, fname="chromaticity_piece_by_localkey")

    return result_df


def compute_piece_chromaticity_by_mode(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    def calculate_max_min_pc(x):
        if len(x) > 0:
            return max(x), min(x)
        else:  # hacking the zero-length all_tones
            return 0, 0

    df["max_WL"], df["min_WL"] = zip(*df["within_label"].apply(calculate_max_min_pc))
    df["max_OL"], df["min_OL"] = zip(*df["out_of_label"].apply(calculate_max_min_pc))

    result_df = df.groupby(['corpus', 'piece', 'localkey_mode'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        globalkey=("globalkey", "first"),
        localkey_mode=("localkey_mode", "first"),

        max_WL=("max_WL", "max"),
        min_WL=("min_WL", "min"),
        max_OL=("max_OL", "max"),
        min_OL=("min_OL", "min"),

        WLC=("WLC", "mean"),
        OLC=("OLC", "mean")
    )

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)
    result_df["WL_5th_range"] = (result_df["max_WL"] - result_df["min_WL"]).abs()
    result_df["OL_5th_range"] = (result_df["max_OL"] - result_df["min_OL"]).abs()

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result_df, file_type="both", directory=dir, fname="chromaticity_piece_by_mode")

    return result_df


# Define functions: Dissonance
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
        ["corpus", "piece", "corpus_year", "piece_year", "globalkey", "localkey",
         "duration_qb_frac", "chord", "within_label", "ICs", "WLD"]]

    if save:
        dir = "../Data/prep_data/for_analysis/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_df(df=result, file_type="both", directory=dir, fname="dissonance_chord")
    return result


def compute_piece_dissonance(df: pd.DataFrame, weighted: bool, save: bool = True) -> pd.DataFrame:
    if weighted:
        # Define a lambda function to compute the dissonance weighted by chord duration:
        wd = lambda x: np.average(x, weights=df.loc[x.index, "duration_qb_frac"])
        print(df.loc[df.index, "duration_qb_frac"])

        result_df_gb = df.groupby(['corpus', 'piece'])
        print(dict(list(result_df_gb)))

    else:
        chord_num = len(df.index)


if __name__ == "__main__":
    # data = process_DLC_data(data_path="../Data/prep_data/DLC_data.pickle", save=True)

    # prep_DLC_data = load_file_as_df(path="../Data/prep_data/processed_DLC_data.pickle")

    # print(f'Computing chord chromaticity indices ...')
    # chord_chromaticity = compute_chord_chromaticity(df=prep_DLC_data)
    #
    # print(f'Computing piece-level chromaticity ...')
    # piece_chromaticity_by_key = compute_piece_chromaticity_by_key_segment(df=chord_chromaticity)
    # piece_chromaticity_by_mode = compute_piece_chromaticity_by_mode(df=chord_chromaticity)

    # print(f'Computing chord dissonance indices ...')
    # chord_dissonance = compute_chord_dissonance(df=prep_DLC_data)
    pd.set_option('display.max_columns', None)

    chord_dis = load_file_as_df(path="../Data/prep_data/for_analysis/dissonance_chord.pickle")
    print(f'Computing piece dissonance indices ...')
    compute_piece_dissonance(df=chord_dis, weighted=True, save=True)
