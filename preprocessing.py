import concurrent.futures
import copy
import os
from fractions import Fraction
from typing import Literal, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from utils.util import setup_logger

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

def _get_unannotated_pieces(metadata: pd.DataFrame) -> List:
    # check the "label_count" column from the metadata tsv
    unannotated_data = metadata[metadata["label_count"] == 0]
    unannotated_corpus_piece_pairs = list(zip(unannotated_data["corpus"], unannotated_data["piece"]))
    return unannotated_corpus_piece_pairs


def filter_unannotated_pieces(df: pd.DataFrame, log_path: str = "data/preprocessing_logs/") -> pd.DataFrame:
    unannotated_logger = setup_logger('unannotated_logger', log_path + f"filter_warning_unannotated.log")
    metadata_df = pd.read_csv("data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.metadata.tsv",
                              sep="\t")

    unannotated_pieces = _get_unannotated_pieces(metadata_df)

    unannotated_logger.warning(f"Unannotated pieces ===============================:")
    for piece in unannotated_pieces:
        unannotated_logger.warning(piece)

    corpus_list, piece_list = zip(*unannotated_pieces)

    # Filter df based on the condition where (corpus, piece) tuples are not in the unannotated_pieces list
    result = df[~(df['corpus'].isin(corpus_list) & df['piece'].isin(piece_list))]
    return result


def filter_df_rows(df: pd.DataFrame,
                   kind: Literal["harmonies", "notes"],
                   log_path: str = "data/preprocessing_logs/") -> pd.DataFrame:
    """remove rows not satisfying the "rules" in the harmony or note dataframe:
     - 'quarterbeats' col with NaN value (they indicate the first ending, excluded from the quarterbeats index)
     - 'chord' col with NaN val (usually the beginning of phrase marking in previous annotation standard)
     """
    print(f"Calling filter_df_rows with kind: {kind}")
    if kind == "harmonies":
        columns_to_log = ["corpus", "piece", "quarterbeats", 'duration_qb', 'chord', 'chord_tones', 'globalkey',
                          'localkey', "root"]
        # Set up warning logs
        h_filter_logger = setup_logger('h_filter_logger', log_path + f"filter_warning_{kind}.log")

        # # check the dataframe, if missing (not annotated, and bad parsing), remove the row and log
        # sub_df = df[
        #     ['quarterbeats', 'duration_qb', 'chord', 'chord_tones', 'globalkey', 'localkey', "root", "bass_note"]]

        # missing_rows = df[
        #     (sub_df.isna().any(axis=1) | (sub_df == '@none').any(axis=1))
        #     | (sub_df['duration_qb'] == 0) | (df['duration_qb'].isna())
        #     ]

        # filtering_rows = df[
        #     (sub_df.isna().any(axis=1) | (sub_df == '@none').any(axis=1))
        #     | (sub_df['duration_qb'] == 0) | (sub_df['duration_qb'].isna())
        #     ]

        filtering_rows = df[
            (df['duration_qb'] == 0) |
            (df[columns_to_log].isnull().any(axis=1)) |
            (df[columns_to_log].isna().any(axis=1)) |
            (df[columns_to_log] == '@none').any(axis=1)
            ]

        subdf = df[['quarterbeats', 'duration_qb', 'globalkey', 'localkey', 'chord', 'chord_tones', 'root']]
        missing_rows = df[
            ((df['duration_qb'] == 0 | (subdf.isna().any(axis=1))))
            & (df[['corpus', 'piece', 'chord']].notnull().all(axis=1))
            ]

        # h_filter_logger.warning(f"Problematic parsing Harmonies df (rows) ===============================:")

        missing_rows[columns_to_log].apply(
            lambda row: h_filter_logger.warning(
                ', '.join([f"{col}: {row[col]}" for col in columns_to_log])
            ),
            axis=1
        )

        # remove rows with missing data:
        df = df[~df.index.isin(filtering_rows.index)].reset_index(drop=True)
        assert not df['duration_qb'].isna().any()

        return df

    elif kind == "notes":
        columns_to_log = ["corpus", "piece", "quarterbeats", 'tpc', 'midi', 'name']

        # Set up warning logs
        n_filter_logger = setup_logger('n_filter_logger', log_path + f"filter_warning_{kind}.log")

        # check parsing consistency of the dataframe
        filtering_rows = df[df[columns_to_log].isnull().any(axis=1)]
        missing_rows = df[
            (df[['quarterbeats', 'tpc', 'midi', 'name']].isnull().any(axis=1))
            & (df[['corpus', 'piece']].notnull().all(axis=1))
            ]

        # n_filter_logger.warning(f"Problematic parsing Notes df (rows) ===============================:")
        missing_rows[columns_to_log].apply(
            lambda row: n_filter_logger.warning(
                ', '.join([f"{col}: {row[col]}" for col in columns_to_log])
            ),
            axis=1
        )
        # remove rows with missing data:
        df = df[~df.index.isin(filtering_rows.index)].reset_index(drop=True)
        assert not df['duration_qb'].isna().any()

        return df
    else:
        raise ValueError


# get a list of piece dfs _____________________________________________________________________________________________

def get_pieces_df_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    # grouped_dfs = {group: group_df for group, group_df in df.groupby(["corpus", "piece"])}
    grouped = df.groupby(["corpus", "piece"])
    df_list = []
    for group_name, group_df in grouped:
        df_list.append(group_df)
    return df_list


# append the pcs column to harmonies df _______________________________________________________________________________

def append_actual_notes_to_harmonies(harmony: pd.DataFrame,
                                     note: pd.DataFrame,
                                     pcs_rep: Literal["tpc", "midi", "name"] = "tpc") -> pd.DataFrame:
    """
    Works for single piece harmonies and notes dfs.
    This function will first filter the harmony and note df before appending.
    """
    # all_tones = all pitches happen in the score

    # 'duration_qb' is a float, which sometimes can be repeating decimals, convert it back to the original fraction
    harmony['duration_qb'] = harmony['duration_qb'].apply(lambda x: Fraction(x).limit_denominator())
    note['duration_qb'] = note['duration_qb'].apply(lambda x: Fraction(x).limit_denominator())

    aligned_dict = align_span_dict(dict_harmony=f(harmony, 'chord', allow_identical_timespan=False),
                                   dict_note=f(note, pcs_rep, allow_identical_timespan=True))

    if pcs_rep == "tpc":
        harmony['tones_in_span_in_C'] = notes_only(aligned_dict)
    elif pcs_rep == "name":
        harmony['all_tones_spc'] = notes_only(aligned_dict)
    else:
        harmony["all_tones_midi"] = notes_only(aligned_dict)

    return harmony


# preprocessing:
def preprocessing_df_Cleaning(metadata: pd.DataFrame,
                              harmonies: pd.DataFrame,
                              notes: pd.DataFrame,
                              save_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata["piece"] = metadata["piece"].str.normalize(form='NFC')  # normalize the strings

    print(f' 2. filtering out unannotated dfs ...')

    harmonies = filter_unannotated_pieces(df=harmonies)
    # use (corpus, piece) in harmonies df to filter out no correspondence rows in notes df (those not annotated)
    harmonies_set = set(map(tuple, harmonies[['corpus', 'piece']].to_records(index=False)))

    def filter_notes_rows(row):
        # function to filter notes df using parallel processing
        return (row.corpus, row.piece) in harmonies_set

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        results = list(executor.map(filter_notes_rows, notes.itertuples(index=False)))

    notes = notes[results]

    print(f' 2.5. filtering out bad parsing (rows) in dfs ...')

    cleaned_harmonies = filter_df_rows(df=harmonies, kind="harmonies").reset_index()
    cleaned_notes = filter_df_rows(df=notes, kind="notes").reset_index()

    cleaned_harmonies.to_csv(f'{save_path}cleaned_harmonies.tsv', sep="\t")
    cleaned_notes.to_csv(f'{save_path}cleaned_notes.tsv', sep="\t")
    print(f"Saved cleaned dfs to {save_path}!")
    return cleaned_harmonies, cleaned_notes


def preprocessing_df_AppendingNotes(metadata: pd.DataFrame,
                                    harmonies: pd.DataFrame,
                                    notes: pd.DataFrame) -> pd.DataFrame:
    print(f' 3. getting piece dfs list ...')
    # get list of pieces dfs:
    harmonies_dfs_list = get_pieces_df_list(df=harmonies)
    notes_dfs_list = get_pieces_df_list(df=notes)

    assert len(harmonies_dfs_list) == len(notes_dfs_list)

    print(f' 4. appending notes to harmony ...')

    # result_harmonies_list = [append_actual_notes_to_harmonies(harmony=hdf, note=ndf, pcs_rep="tpc") for hdf, ndf in
    #                          zip(harmonies_dfs_list, notes_dfs_list)]
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
    final_df = final_df.drop(columns=['Unnamed: 0', 'index'])

    return final_df


if __name__ == "__main__":
    # harmonies_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "globalkey", "localkey",
    #                   "chord", "root", "bass_note", "chord_tones", "added_tones"]
    # notes_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "tpc", 'midi', 'name']
    #
    # print(f' 1. loading dfs ...')
    metadata = pd.read_csv("data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.metadata.tsv",
                           sep="\t")
    # harmonies = pd.read_csv("data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.expanded.tsv",
    #                         sep="\t", usecols=harmonies_cols)
    # notes = pd.read_csv("data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.notes.tsv", sep="\t",
    #                     usecols=notes_cols)
    #
    # clean_harmonies, clean_notes = preprocessing_df_Cleaning(metadata=metadata,
    #                                                            harmonies=harmonies,
    #                                                            notes=notes,
    #                                                            save_path="data/preprocessing_logs/")

    clean_harmonies = pd.read_csv("data/preprocessing_logs/cleaned_harmonies.tsv", sep="\t")
    clean_notes = pd.read_csv("data/preprocessing_logs/cleaned_notes.tsv", sep="\t")
    final_df = preprocessing_df_AppendingNotes(metadata=metadata, harmonies=clean_harmonies, notes=clean_notes)

    path = "data/DLC_harmonies.tsv"
    print(f' 7. saving df to {path} ...')
    final_df.to_csv(path, sep="\t", index=True)
