import ast
from typing import Literal, Optional

import pandas as pd

from Code.utils.auxiliary import determine_period, create_results_folder
import os

from pitchtypes import SpelledPitchClass

from Code.utils.htypes import Numeral, Key
from Code.metrics import cumulative_distance_to_diatonic_set, tpcs_to_ics, pcs_to_dissonance_score
from Code.utils.util import flatten, flatten_to_list, safe_literal_eval, save_df, load_file_as_df

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'


# %% Define function: Loading with additional preprocessing steps and saving

def process_DLC_data(data_path: str,
                     repo_dir: str,
                     save: bool = True) -> pd.DataFrame:
    """
    An intermediate step for loading and process the filtered dcml harmonies tsv before computing chromaticity
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

    df["period_Johannes"] = df.apply(lambda row: determine_period(row=row, method="Johannes"), axis=1)
    df["period_Fabian"] = df.apply(lambda row: determine_period(row=row, method="Fabian"), axis=1)

    try:
        df = df.drop(columns=['Unnamed: 0', 'all_tones_tpc_in_C'])
    except KeyError:
        pass

    if save:
        print(f'Saving the processed DLC data ...')
        save_df(df=df, file_type="both", fname="processed_DLC_data", directory=f"{repo_dir}Data/prep_data/")

    return df


# %% Compute metrics : Chromaticity

def compute_chord_chromaticity(df: pd.DataFrame,
                               repo_dir: str,
                               save: bool = True) -> pd.DataFrame:
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
        ["corpus", "piece", "corpus_year", "piece_year", "period_Johannes", "period_Fabian", "globalkey", "localkey",
         "localkey_spc", "localkey_mode", "quarterbeats", "chord", "tones_in_span_in_C", "tones_in_span_in_lk",
         "within_label", "WLC", "out_of_label", "OLC"]]
    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        save_df(df=result, file_type="both", directory=folder_path, fname="chromaticity_chord")

    return result


def compute_piece_chromaticity(df: pd.DataFrame,
                               by: Literal["key_segment", "mode"],
                               repo_dir: str,
                               save: bool = True) -> pd.DataFrame:
    if by == "key_segment":
        fname = f'chromaticity_piece_by_localkey'
        adj_check = (df.localkey != df.localkey.shift()).cumsum()
        result_df = df.groupby(['corpus', 'piece', adj_check], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),
            localkey_mode=("localkey_mode", "first"),

            WLC=("WLC", "mean"),
            OLC=("OLC", "mean")
        )

    elif by == "mode":
        fname = f'chromaticity_piece_by_mode'

        result_df = df.groupby(['corpus', 'piece', 'localkey_mode'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey_mode=("localkey_mode", "first"),

            WLC=("WLC", "mean"),
            OLC=("OLC", "mean"),

        )
    else:
        raise ValueError()

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        save_df(df=result_df, file_type="both", directory=folder_path, fname=fname)

    return result_df


def get_piece_chromaticity_by_mode_seg(df: pd.DataFrame,
                                       mode: Literal["major", "minor"],
                                       repo_dir: str,
                                       save: bool = True) -> pd.DataFrame:
    """
    df: the piece_chromaticity df by mode, containing the col named "localkey_mode"
    """
    result_df = df.loc[df['localkey_mode'] == mode]

    def calculate_WLC_percentage(row):
        denominator = (row["WLC"] + row["OLC"])
        if denominator != 0:
            return row["WLC"] / denominator
        else:
            return 0

    def calculate_OLC_percentage(row):
        denominator = (row["WLC"] + row["OLC"])
        if denominator != 0:
            return row["OLC"] / denominator
        else:
            return 0

    result_df["WLC_percentage"] = result_df.apply(lambda row: calculate_WLC_percentage(row=row), axis=1)
    result_df["OLC_percentage"] = result_df.apply(lambda row: calculate_OLC_percentage(row=row), axis=1)

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        fname = f'chromaticity_piece_{mode}'
        save_df(df=result_df, file_type="both", directory=folder_path, fname=fname)
    return result_df


# %% Compute metrics : Dissonance
def compute_chord_dissonance(df: pd.DataFrame,
                             repo_dir: str,
                             save: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------

    df : pd.DataFrame
        should be the processed_DLC_data
    """

    # WITHIN-LABEL DISSONANCE
    df["interval_classes"] = df.apply(lambda row: tpcs_to_ics(tpcs=row["within_label"]), axis=1)
    df["WLD"] = df.apply(lambda row: pcs_to_dissonance_score(tpcs=row["within_label"]), axis=1)

    result = df[
        ["corpus", "piece", "corpus_year", "piece_year", "period_Johannes", "period_Fabian", "globalkey", "localkey",
         "localkey_mode", "duration_qb_frac", "quarterbeats", "chord", "within_label", "interval_classes", "WLD"]]

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        save_df(df=result, file_type="both", directory=folder_path, fname="dissonance_chord")
    return result


def compute_piece_dissonance(df: pd.DataFrame,
                             by: Optional[Literal["key_segment", "mode"]],
                             repo_dir: str,
                             save: bool = True) -> pd.DataFrame:
    if by == "key_segment":
        fname_suffix = f'_by_localkey'
        chord_num_df = df.groupby(['corpus', 'piece', 'localkey']).size().rename("chord_num").to_frame()
        adj_check = (df.localkey != df.localkey.shift()).cumsum()
        grouped_df = df.groupby(['corpus', 'piece', adj_check], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey=("localkey", "first"),
            localkey_mode=("localkey_mode", "first"),
            total_WLD=("WLD", "sum")
        )
        res_df = pd.merge(grouped_df, chord_num_df, on=['corpus', 'piece', 'localkey'], how='left')

    elif by == "mode":
        fname_suffix = f'_by_mode'
        chord_num_df = df.groupby(['corpus', 'piece', 'localkey_mode']).size().rename("chord_num").to_frame()
        grouped_df = df.groupby(['corpus', 'piece', 'localkey_mode'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey_mode=("localkey_mode", "first"),
            total_WLD=("WLD", "sum")
        )
        res_df = pd.merge(grouped_df, chord_num_df, on=['corpus', 'piece', 'localkey_mode'], how='left')

    elif by is None:
        fname_suffix = f''
        chord_num_df = df.groupby(['corpus', 'piece']).size().rename("chord_num").to_frame()
        grouped_df = df.groupby(['corpus', 'piece'], as_index=False, sort=False).agg(
            corpus_year=("corpus_year", "first"),
            piece_year=("piece_year", "first"),
            period_Fabian=("period_Fabian", "first"),
            period_Johannes=("period_Johannes", "first"),
            globalkey=("globalkey", "first"),
            localkey_mode=("localkey_mode", "first"),
            total_WLD=("WLD", "sum"))
        res_df = pd.merge(grouped_df, chord_num_df, on=['corpus', 'piece'], how='left')
    else:
        raise ValueError

    res_df["avg_WLD"] = res_df.total_WLD / res_df.chord_num

    res_df["corpus_id"] = pd.factorize(res_df["corpus"])[0] + 1
    res_df["piece_id"] = pd.factorize(res_df["piece"])[0] + 1

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        fname = f"dissonance_piece{fname_suffix}"
        save_df(df=res_df, file_type="both", directory=folder_path, fname=fname)
    return res_df


# %% Compute metric: fifths range

def compute_piece_fifth_range(df: pd.DataFrame,
                              repo_dir: str,
                              save: bool = True) -> pd.DataFrame:
    """
    df: taking any chord-level df containing the columns "within_label" and out_of_label" => ("chromaticity_chord")
    """

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
        period_Fabian=("period_Fabian", "first"),
        period_Johannes=("period_Johannes", "first"),
        globalkey=("globalkey", "first"),
        localkey_mode=("localkey_mode", "first"),

        max_WL=("max_WL", "max"),
        min_WL=("min_WL", "min"),
        max_OL=("max_OL", "max"),
        min_OL=("min_OL", "min")
    )

    result_df = result_df.sort_values(by=["corpus_year", "piece_year"], ignore_index=True)
    result_df["WL_5th_range"] = (result_df["max_WL"] - result_df["min_WL"]).abs()
    result_df["OL_5th_range"] = (result_df["max_OL"] - result_df["min_OL"]).abs()

    result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    result_df["piece_id"] = pd.factorize(result_df["piece"])[0] + 1

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        fname = f'fifths_range_piece'
        save_df(df=result_df, file_type="both", directory=folder_path, fname=fname)

    return result_df


# %% Combine metric dataframes
def combine_chord_level_indices(chord_chromaticity: pd.DataFrame,
                                chord_dissonance: pd.DataFrame,
                                repo_dir: str,
                                save: bool = True) -> pd.DataFrame:
    common_cols = ['corpus', 'piece', "corpus_year", "piece_year", "period_Johannes", "period_Fabian", "globalkey",
                   "localkey", "localkey_mode", "quarterbeats",
                   "chord", "within_label"]

    chromaticity = chord_chromaticity[common_cols + ["out_of_label", "WLC", "OLC"]]
    dissonance = chord_dissonance[common_cols + ["interval_classes", "WLD"]]

    chromaticity.within_label = chromaticity.within_label.apply(tuple)
    chromaticity.out_of_label = chromaticity.out_of_label.apply(tuple)
    dissonance.within_label = dissonance.within_label.apply(tuple)
    dissonance.interval_classes = dissonance.interval_classes.apply(tuple)

    result_df = chromaticity.merge(dissonance,
                                   on=common_cols,
                                   how="outer")[common_cols + ["out_of_label", "interval_classes", "WLC", "OLC", "WLD"]]

    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)

        save_df(df=result_df, file_type="both", directory=folder_path, fname="chord_level_indices")

    return result_df


def combined_piece_level_indices(piece_chrom: pd.DataFrame,
                                 piece_diss: pd.DataFrame,
                                 groupy_by: Literal["key_segment", "mode"],
                                 repo_dir: str) -> pd.DataFrame:
    # common_cols = ['corpus', 'piece', "corpus_year", "piece_year", "period_Johannes", "period_Fabian", "globalkey"]

    if groupy_by == "mode":
        fname_suffix = f'_by_mode'
        common_cols = ['corpus', 'piece', "corpus_id", "piece_id", "corpus_year", "piece_year", "period_Johannes",
                       "period_Fabian", "globalkey",
                       "localkey_mode"]


    elif groupy_by == "key_segment":
        fname_suffix = f'_by_localkey'
        common_cols = ['corpus', 'piece', "corpus_id", "piece_id", "corpus_year", "piece_year", "period_Johannes",
                       "period_Fabian", "globalkey",
                       "localkey"]

    else:
        raise ValueError()

    chromaticity = piece_chrom[common_cols + ["WLC", "OLC"]]
    WLD = piece_diss["avg_WLD"].tolist()

    assert chromaticity.shape[0] == len(WLD)

    res_df = chromaticity.assign(WLD=WLD)

    # save data:
    fname = f'piece_level_indices{fname_suffix}'
    folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
    save_df(df=res_df, file_type="both", directory=folder_path, fname=fname)

    return res_df


def get_corpora_level_indices_by_mode(df: pd.DataFrame,
                                      repo_dir: str) -> pd.DataFrame:
    """
    df: piece_indices_by_mode_df
    """

    result_df = df.groupby(['corpus', 'localkey_mode'], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),

        localkey_mode=("localkey_mode", "first"),
        corpus_id=("corpus_id", "first"),

        WLC=("WLC", "mean"),
        OLC=("OLC", "mean"),
        WLD=("WLD", "mean")

    )
    folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
    fname = f'corpora_level_indices_by_mode'
    save_df(df=result_df, file_type="both", directory=folder_path, fname=fname)
    return result_df


# %% Chord-level indices correlation

def compute_pairwise_chord_indices_r_by_piece(df: pd.DataFrame,
                                              repo_dir: str,
                                              save: bool) -> pd.DataFrame:
    """
    df: assuming we take the chord-level indices
    """
    piece_df = df.groupby(["corpus", "piece", "localkey_mode"], as_index=False, sort=False).agg(
        corpus_year=("corpus_year", "first"),
        piece_year=("piece_year", "first"),
        period_Fabian=("period_Fabian", "first"),
        period_Johannes=("period_Johannes", "first"),
        globalkey=("globalkey", "first"))

    r_wlc_olc = df.groupby(["corpus", "piece", "localkey_mode"], sort=False)[['WLC', 'OLC']].corr().unstack().iloc[:,
                1].reset_index(
        name=f'r_WLC_OLC')

    r_wlc_wld = df.groupby(["corpus", "piece", "localkey_mode"], sort=False)[['WLC', 'WLD']].corr().unstack().iloc[:,
                1].reset_index(
        name=f'r_WLC_WLD')

    r_df = r_wlc_olc.merge(r_wlc_wld,
                           on=["corpus", "piece", "localkey_mode"],
                           how="outer")

    res_df = piece_df.merge(r_df, on=["corpus", "piece", "localkey_mode"])

    res_df = res_df.fillna(0)

    res_df["corpus_id"] = pd.factorize(res_df["corpus"])[0] + 1
    res_df["piece_id"] = pd.factorize(res_df["piece"])[0] + 1

    # save df:
    if save:
        folder_path = create_results_folder(parent_folder="Data", analysis_name=None, repo_dir=repo_dir)
        fname = f"chord_indices_r_vals_by_piece"
        save_df(df=res_df, file_type="both", directory=folder_path, fname=fname)

    return res_df


# %% full set of processed datasets for analyses
def full_post_preprocessed_datasets_update():
    user = os.path.expanduser("~")
    repo_dir = f'{user}/Codes/chromaticism-codes/'
    prep_DLC_data = load_file_as_df(path=f"{repo_dir}Data/prep_data/processed_DLC_data.pickle")

    print(f'Computing chord chromaticity indices ...')
    chord_chrom = compute_chord_chromaticity(df=prep_DLC_data, repo_dir=repo_dir)

    print(f'Computing piece-level chromaticity ...')
    piece_chrom_by_localkey = compute_piece_chromaticity(df=chord_chrom, by="key_segment", repo_dir=repo_dir)
    piece_chrom_by_mode = compute_piece_chromaticity(df=chord_chrom, by="mode", repo_dir=repo_dir)

    print(f'    Getting separate piece-level chromaticties for major/minor mode segments...')

    piece_chrom_by_mode = load_file_as_df(
        "/Users/xguan/Codes/chromaticism-codes/Data/prep_data/for_analysis/chromaticity_piece_by_mode.pickle")
    get_piece_chromaticity_by_mode_seg(df=piece_chrom_by_mode, mode="major", repo_dir=repo_dir)
    get_piece_chromaticity_by_mode_seg(df=piece_chrom_by_mode, mode="minor", repo_dir=repo_dir)

    print(f'Finished chromaticity!')

    print(f'Computing chord dissonance indices ...')
    chord_dissonance = compute_chord_dissonance(df=prep_DLC_data, repo_dir=repo_dir)

    print(f'Computing piece dissonance indices ...')
    piece_diss_by_mode = compute_piece_dissonance(by="mode", df=chord_dissonance, repo_dir=repo_dir, save=True)
    piece_diss_by_lk = compute_piece_dissonance(by="key_segment", df=chord_dissonance, repo_dir=repo_dir, save=True)
    piece_diss = compute_piece_dissonance(by=None, df=chord_dissonance, repo_dir=repo_dir, save=True)

    print(f'Finished dissonance!')
    #
    chord_chrom = load_file_as_df(path=f"{repo_dir}Data/prep_data/for_analysis/chromaticity_chord.pickle")
    chord_diss = load_file_as_df(path=f"{repo_dir}Data/prep_data/for_analysis/dissonance_chord.pickle")

    print(f'Combing dfs for chord-level chromaticity and dissonance metrics ...')
    chord_df = combine_chord_level_indices(chord_chromaticity=chord_chrom, chord_dissonance=chord_diss,
                                           repo_dir=repo_dir)
    print(f'   computing pairwise correlation for chord indices...')
    compute_pairwise_chord_indices_r_by_piece(df=chord_df, repo_dir=repo_dir, save=True)

    print(f'Combing dfs for piece-level indices ...')
    piece_by_mode = combined_piece_level_indices(piece_chrom=piece_chrom_by_mode, piece_diss=piece_diss_by_mode,
                                                 repo_dir=repo_dir, groupy_by="mode")
    piece_by_lk = combined_piece_level_indices(piece_chrom=piece_chrom_by_localkey, piece_diss=piece_diss_by_lk,
                                               repo_dir=repo_dir, groupy_by="key_segment")

    print(f'Computing corpora-level indices ... ')
    get_corpora_level_indices_by_mode(df=piece_by_mode, repo_dir=repo_dir)

    print(f'Fini!')


if __name__ == "__main__":
    full_post_preprocessed_datasets_update()

    # user = os.path.expanduser("~")
    # repo_dir = f'{user}/Codes/chromaticism-codes/'

    # process_DLC_data(data_path=f"{repo_dir}Data/prep_data/DLC_data.pickle", save=True, repo_dir=repo_dir)
    # prep_DLC_data = load_file_as_df(path=f"{repo_dir}Data/prep_data/processed_DLC_data.pickle")
