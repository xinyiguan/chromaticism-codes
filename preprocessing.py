import ast
import copy
from fractions import Fraction
import os

from typing import Literal, List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from pitchtypes import SpelledPitchClass

from utils.htypes import Numeral, Key
from utils.util import setup_logger, flatten, safe_literal_eval, flatten_to_list
import concurrent.futures

Note = str
Harmony = ...
Span = Tuple[Fraction, Fraction]


# get aligned pcvs and chord label ____________________________________________________________________________________
def within(x: Span, y: Span) -> bool:
    """check if the two spans have temporal overlap (duration > 0)"""
    has_temporal_overlap = not (x[1] <= y[0] or y[1] <= x[0])
    return has_temporal_overlap


def align_span_dict(dict_note: Dict[Span, List[Note]],
                    dict_harmony: Dict[Span, Harmony]
                    ) -> Dict[Span, Tuple[Harmony, List[Note]]]:
    big_spans = dict_harmony.keys()
    small_spans = dict_note.keys()

    compatible_span_pairs = []
    remaining_small_spans = copy.copy(list(small_spans))

    for S in big_spans:
        compatible_small_spans = []
        for i, s in enumerate(remaining_small_spans):
            if within(s, S):
                compatible_small_spans.append(s)

        compatible_span_pairs.append((S, compatible_small_spans))
        for x in compatible_small_spans:
            remaining_small_spans.remove(x)

    joined_dict = {big_span: (dict_harmony[big_span], [dict_note[span] for span in small_spans]) for
                   (big_span, small_spans) in compatible_span_pairs}
    return joined_dict


def f(df: pd.DataFrame, attribute: str, allow_identical_timespan: bool = False) -> Dict[Span, Any]:
    """extracts temporal spans (represented as pairs of start and end times) from the DataFrame
    based on the 'quarterbeats' and 'duration_qb' columns."""

    # 'duration_qb' is a float, which sometimes can be repeating decimals, convert it back to the original fraction
    df['duration_qb_frac'] = df['duration_qb'].apply(lambda x: Fraction(x).limit_denominator())

    spans = [(Fraction(b), Fraction(b) + Fraction(d)) for b, d in
             df[['quarterbeats', 'duration_qb_frac']].itertuples(index=False, name=None)]
    if allow_identical_timespan:
        result = {}
        for s, x in zip(spans, df[attribute]):
            if s in result:
                result[s].append(x)
            else:
                result[s] = [x]
        # result = {s: [df[attribute][i] for ] for s in spans}
    else:
        result = {s: x for s, x in zip(spans, df[attribute])}
    return result


def notes_only(d: Dict[Span, Tuple[Harmony, List[Note]]]) -> List[List[Note]]:
    notes_column = [p[1] for p in d.values()]
    return notes_column


# filtering ____________________________________________________________________________________________________________

def filter_df(df: pd.DataFrame, kind: Literal["harmonies", "notes"],
              log_path: str = "data/") -> pd.DataFrame:
    """remove rows not satisfying the "rules" in the harmonies and notes dataframes:
     - 'quarterbeats' col with NaN value (they indicate the first ending, excluded from the quarterbeats index)
     - 'chord' col with NaN val (usually the beginning of phrase marking in previous annotation standard)
     """
    # Set up warning logs
    filter_logger = setup_logger('filter_logger', log_path + f"filter_warning_{kind}.log")

    if kind == "harmonies":
        # check the dataframe, if missing (not annotated, and bad parsing), remove the row and log
        sub_df = df[
            ['quarterbeats', 'duration_qb', 'chord', 'chord_tones', 'globalkey', 'localkey', "root", "bass_note"]]
        missing_rows = df[
            (sub_df.isna().any(axis=1) | (sub_df == '@none').any(axis=1))
            | (sub_df['duration_qb'] == 0) | (df['duration_qb'].isna())
            ]

        # additional check on duration_qb # TODO: remove condition after correctly fixing the github issue

    elif kind == "notes":
        # check parsing consistency of the dataframe
        missing_rows = df[df[['quarterbeats', 'tpc', 'midi', 'name']].isnull().any(axis=1)]

    else:
        raise ValueError

    missing_rows_indices = missing_rows.index.to_list()
    if missing_rows_indices:  # Check if the list is not empty
        filter_logger.warning(f"missing rows indices \n{missing_rows_indices}\n")
    missing_rows.apply(
        lambda row: filter_logger.warning(f"\n{row}\n"),
        axis=1)

    # remove rows with missing data:
    df = df[~df.index.isin(missing_rows.index)].reset_index(drop=True)
    assert not df['duration_qb'].isna().any()

    return df


# get a list of piece dfs _____________________________________________________________________________________________

def get_pieces_df_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    # grouped_dfs = {group: group_df for group, group_df in df.groupby(["corpus", "piece"])}
    grouped = df.groupby(["corpus", "piece"])
    df_list = []
    for group_name, group_df in grouped:
        df_list.append(group_df)
    return df_list


# append the pcs column to harmonies df _______________________________________________________________________________

def append_actual_notes_to_harmonies(harmony: pd.DataFrame, note: pd.DataFrame,
                                     pcs_rep: Literal["tpc", "midi", "name"]) -> pd.DataFrame:
    """
    Works for single piece harmonies and notes dfs.
    This function will first filter the harmony and note df before appending.
    """
    # all_tones = all pitches happen in the score
    filtered_harmony = filter_df(df=harmony, kind="harmonies", log_path="old_data/")
    filtered_note = filter_df(df=note, kind="notes", log_path="old_data/")

    # 'duration_qb' is a float, which sometimes can be repeating decimals, convert it back to the original fraction
    harmony['duration_qb'] = filtered_harmony['duration_qb'].apply(lambda x: Fraction(x).limit_denominator())
    note['duration_qb'] = filtered_note['duration_qb'].apply(lambda x: Fraction(x).limit_denominator())

    aligned_dict = align_span_dict(dict_harmony=f(filtered_harmony, 'chord', allow_identical_timespan=False),
                                   dict_note=f(filtered_note, pcs_rep, allow_identical_timespan=True))

    if pcs_rep == "tpc":
        filtered_harmony['tones_in_span_in_C'] = notes_only(aligned_dict)
    elif pcs_rep == "name":
        filtered_harmony['all_tones_spc'] = notes_only(aligned_dict)
    else:
        filtered_harmony["all_tones_midi"] = notes_only(aligned_dict)

    return filtered_harmony


# append additional cols ______________________________________________________________________________________________

def append_additional_cols(hdf: pd.DataFrame, mdf: pd.DataFrame) -> pd.DataFrame:
    # helper funcs
    def find_lk_spc(row):
        """get the local key tonic (in roman numeral) in spelled pitch class"""
        return Numeral.from_string(s=row["localkey"], k=Key.from_string(s=row["globalkey"])).key_if_tonicized().tonic

    def lk2c_dist(row):
        """get the fifths distance from the local key tonic to C"""
        return row["localkey_spc"].interval_to(SpelledPitchClass("C")).fifths()

    def correct_tpc_ref_center(row):
        return [x + row["lk2C"] for x in row["tones_in_span_in_C"]]

    def tones_not_in_chordtones(row):
        return [item for item in row['tones_in_span'] if item not in row['all_chordtones']]

    hdf["localkey_spc"] = hdf.apply(find_lk_spc, axis=1)

    hdf["lk2C"] = hdf.apply(lk2c_dist, axis=1)

    # flatten and get unique tpc in the list in all_tones col
    hdf["all_tones_tpc_in_C"] = hdf["all_tones_tpc_in_C"].apply(lambda s: list(ast.literal_eval(s))).apply(
        lambda lst: list(flatten(lst))).apply(lambda l: list(set(l)))

    # correct the tpc to reference to local key tonic
    hdf["tones_in_span"] = hdf.apply(correct_tpc_ref_center, axis=1)

    hdf["added_tones"] = hdf["added_tones"].apply(lambda s: safe_literal_eval(s)).apply(flatten_to_list)
    hdf["chord_tones"] = hdf["chord_tones"].apply(lambda s: list(ast.literal_eval(s)))
    hdf["all_chordtones"] = hdf["chord_tones"] + hdf[
        "added_tones"]

    hdf["nonchordtones"] = hdf.apply(tones_not_in_chordtones, axis=1)

    # Create a dict mapping from (corpus, piece) to year from metadata_df
    mapping = dict(zip(zip(mdf['corpus'], mdf['piece']), mdf['composed_end']))

    # add the year info
    hdf["piece_year"] = hdf.apply(lambda row: mapping[(row['corpus'], row['piece'])], axis=1)
    hdf = hdf.assign(corpus_year=hdf.groupby("corpus")["piece_year"].transform(np.mean)).sort_values(
        ['corpus_year', 'piece_year']).reset_index(drop=True)

    return hdf


if __name__ == "__main__":
    harmonies_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "globalkey", "localkey",
                      "chord", "root", "bass_note", "chord_tones", "added_tones"]

    notes_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "tpc", 'midi', 'name']

    print(f' 1. loading dfs ...')

    metadata = pd.read_csv("old_data/all_subcorpora/all_subcorpora.metadata.tsv", sep="\t")
    metadata["piece"] = metadata["piece"].str.normalize(form='NFC')  # normalize the strings

    harmonies = pd.read_csv("old_data/all_subcorpora/all_subcorpora.expanded.tsv", sep="\t", usecols=harmonies_cols)
    notes = pd.read_csv("old_data/all_subcorpora/all_subcorpora.notes.tsv", sep="\t", usecols=notes_cols)

    print(f' 2. filtering out notes df ...')
    # use (corpus, piece) in harmonies df to filter out no correspondence rows in notes df (those not annotated)
    harmonies_set = set(map(tuple, harmonies[['corpus', 'piece']].to_records(index=False)))


    def filter_notes_rows(row):
        # function to filter notes df using parallel processing
        return (row.corpus, row.piece) in harmonies_set


    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        results = list(executor.map(filter_notes_rows, notes.itertuples(index=False)))

    notes = notes[results]

    print(f' 3. getting piece dfs list ...')
    # get list of pieces dfs:
    harmonies_dfs_list = get_pieces_df_list(df=harmonies)
    notes_dfs_list = get_pieces_df_list(df=notes)

    assert len(harmonies_dfs_list) == len(notes_dfs_list)

    print(f' 4. appending notes to harmony ...')

    def process_harmonies(hdf: pd.DataFrame, ndf: pd.DataFrame):
        # function to append notes to harmonies df using parallel processing
        return append_actual_notes_to_harmonies(harmony=hdf, note=ndf, pcs_rep="tpc")


    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        result_harmonies_list = [executor.submit(process_harmonies, hdf, ndf).result() for hdf, ndf in
                                 zip(harmonies_dfs_list, notes_dfs_list)]

    print(f' 5. concatenating dfs ...')

    harmonies = pd.concat(result_harmonies_list, ignore_index=True)

    print(f' 6. adding metadata to the df ...')
    # Create a dict mapping from (corpus, piece) to year from metadata_df
    mapping = dict(zip(zip(metadata['corpus'], metadata['piece']), metadata['composed_end']))

    # add the year info
    harmonies["piece_year"] = harmonies.apply(lambda row: mapping[(row['corpus'], row['piece'])], axis=1)
    final_df = harmonies.assign(corpus_year=harmonies.groupby("corpus")["piece_year"].transform(np.mean)).sort_values(
        ['corpus_year', 'piece_year']).reset_index(drop=True)

    # print(f' 6. adding more relevant columns to the df ...')
    # # add additional cols
    # final_df = append_additional_cols(harmonies, metadata)

    path = "old_data/dcml_harmonies.tsv"
    print(f' 7. saving df to {path} ...')
    final_df.to_csv(path, sep="\t", index=True)
