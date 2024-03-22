import concurrent.futures
import copy
import fractions
import os
from fractions import Fraction
from typing import Literal, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from Code.utils.util import setup_logger, DTYPES, load_tsv_as_df, str2inttuple, int2bool

Note = str
Harmony = ...
Span = Tuple[Fraction, Fraction]


# %% Define functions: get aligned pcs and chord label
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
        harmony['all_tones_tpc_in_C'] = notes_only(aligned_dict)
    elif pcs_rep == "name":
        harmony['all_tones_spc'] = notes_only(aligned_dict)
    else:
        harmony["all_tones_midi"] = notes_only(aligned_dict)

    return harmony


# %% Define functions:  filtering

def _get_unannotated_pieces(metadata: pd.DataFrame) -> List:
    # check the "label_count" column from the metadata tsv
    unannotated_data = metadata[metadata["label_count"] == 0]
    unannotated_corpus_piece_pairs = list(zip(unannotated_data["corpus"], unannotated_data["piece"]))
    return unannotated_corpus_piece_pairs


def filter_unannotated_pieces(corpus_name: str,
                              additional_pieces_to_exclude: List[Tuple[str, str]],
                              log_path: str = "../Data/preprocess_logs/") -> pd.DataFrame:
    """
    corpus_name: the name of the corpus
    additional_pieces_to_exclue: format (CORPUS_NAME, PIECE_NAME)
                                (e.g., [("frescobaldi_fiori_musicali", "12.45_Toccata_per_l'Elevatione")])
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    unannotated_logger = setup_logger('unannotated_logger', log_path + f"filter_warning_unannotated.log")

    metadata, harmonies = load_dfs_from_corpus(corpus_name=corpus_name, df_type=["metadata", "harmonies"])

    unannotated_pieces = _get_unannotated_pieces(metadata)
    exclude_pieces = unannotated_pieces + additional_pieces_to_exclude

    unannotated_logger.warning(f"Unannotated pieces ===============================:")
    for piece in exclude_pieces:
        unannotated_logger.warning(piece)

    corpus_list, piece_list = zip(*exclude_pieces)

    # Filter df based on the condition where (corpus, piece) tuples are not in the unannotated_pieces list
    result = harmonies[~(harmonies['corpus'].isin(corpus_list) & harmonies['piece'].isin(piece_list))]
    return result


def filter_df_rows(df: pd.DataFrame,
                   kind: Literal["harmonies", "notes"],
                   log_path: str = "../Data/preprocess_logs/") -> pd.DataFrame:
    """remove rows not satisfying the "rules" in the harmony or note dataframe:
     - 'quarterbeats' col with NaN value (they indicate the first ending, excluded from the quarterbeats index)
     - 'chord' col with NaN val (usually the beginning of phrase marking in previous annotation standard)
     """
    print(f"Calling filter_df_rows with kind: {kind}")
    if kind == "harmonies":
        columns_to_log = ["corpus", "piece", "quarterbeats", 'duration_qb', 'chord', 'chord_tones', 'globalkey',
                          'localkey', "root"]
        # Set up warning logs
        h_filter_logger = setup_logger('harmonies_filter_logger', log_path + f"filter_warning_{kind}.log")

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
        n_filter_logger = setup_logger('notes_filter_logger', log_path + f"filter_warning_{kind}.log")

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


# %% Define functions: preprocessing steps

def get_pieces_df_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    # grouped_dfs = {group: group_df for group, group_df in df.groupby(["corpus", "piece"])}
    grouped = df.groupby(["corpus", "piece"])
    df_list = []
    for group_name, group_df in grouped:
        df_list.append(group_df)
    return df_list


def load_dfs_from_corpus(corpus_name: str,
                         df_type: List[Literal["metadata", "harmonies", "notes"]],
                         Data_path: str = "../Data/") -> Tuple:
    CONVERTERS = {
        'added_tones': str2inttuple,
        'chord_tones': str2inttuple,
        'duration': fractions.Fraction,
        'globalkey_is_minor': int2bool,
        'localkey_is_minor': int2bool,
    }
    harmonies_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "globalkey", "localkey", "localkey_is_minor",
                      "chord", "root", "bass_note", "chord_tones", "added_tones"]
    notes_cols = ["corpus", "piece", "quarterbeats", "duration_qb", "tpc", 'midi', 'name']

    metadata = pd.read_csv(f'{Data_path}{corpus_name}/{corpus_name}.metadata.tsv', sep="\t", engine='python')
    harmonies = pd.read_csv(f'{Data_path}{corpus_name}/{corpus_name}.expanded.tsv', sep="\t", engine='python',
                            usecols=harmonies_cols, dtype=DTYPES, converters=CONVERTERS)
    notes = pd.read_csv(f'{Data_path}{corpus_name}/{corpus_name}.notes.tsv', sep="\t", engine='python',
                        usecols=notes_cols, dtype=DTYPES, converters=CONVERTERS)

    metadata["piece"] = metadata["piece"].str.normalize(form='NFC')  # normalize the strings

    dict_map = {'metadata': metadata, 'harmonies': harmonies, 'notes': notes}

    result = tuple([dict_map[t] for t in df_type])

    return result


def preprocess_df_Cleaning(corpus_name: str,
                           additional_pieces_to_exclude: List[Tuple[str, str]],
                           save_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f'Preprocess Step 1: Loading dfs ...')
    notes: pd.DataFrame = load_dfs_from_corpus(corpus_name=corpus_name, df_type=["notes"])[0]

    print(f'Preprocess Step 2: Filtering out unannotated dfs ...')

    harmonies = filter_unannotated_pieces(corpus_name=corpus_name,
                                          additional_pieces_to_exclude=additional_pieces_to_exclude)

    # use (corpus, piece) in harmonies df to filter out no correspondence rows in notes df (those not annotated)
    harmonies_set = set(map(tuple, harmonies[['corpus', 'piece']].to_records(index=False)))

    def filter_notes_rows(row):
        # function to filter notes df using parallel processing
        return (row.corpus, row.piece) in harmonies_set

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        results = list(executor.map(filter_notes_rows, notes.itertuples(index=False)))

    notes = notes[results]

    print(f'Preprocess Step 2.5: Filtering out bad parsing (rows) in dfs ...')

    cleaned_harmonies = filter_df_rows(df=harmonies, kind="harmonies")
    cleaned_notes = filter_df_rows(df=notes, kind="notes")

    cleaned_harmonies.to_csv(f'{save_path}cleaned_{corpus_name}_harmonies.tsv', sep="\t")
    cleaned_notes.to_csv(f'{save_path}cleaned_{corpus_name}_notes.tsv', sep="\t")
    print(f"Saved cleaned dfs to {save_path}!")
    return cleaned_harmonies, cleaned_notes


def preprocess_df_AppendingNotes(metadata: pd.DataFrame,
                                 harmonies: pd.DataFrame,
                                 notes: pd.DataFrame,
                                 save_df_name: str,
                                 save_path: str) -> pd.DataFrame:
    print(f'Preprocess Step 3: Getting piece dfs list ...')
    # get list of pieces dfs:
    harmonies_dfs_list = get_pieces_df_list(df=harmonies)
    notes_dfs_list = get_pieces_df_list(df=notes)

    assert len(harmonies_dfs_list) == len(notes_dfs_list)

    print(f'Preprocess Step 4: Appending notes to harmony ...')

    def process_harmonies(hdf: pd.DataFrame, ndf: pd.DataFrame):
        # function to append notes to harmonies df using parallel processing
        return append_actual_notes_to_harmonies(harmony=hdf, note=ndf, pcs_rep="tpc")

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        result_harmonies_list = [executor.submit(process_harmonies, hdf, ndf).result() for hdf, ndf in
                                 zip(harmonies_dfs_list, notes_dfs_list)]

    print(f'Preprocess Step 5: Concatenating dfs ...')

    harmonies = pd.concat(result_harmonies_list, ignore_index=True)

    print(f'Preprocess Step 6: Adding metadata to the df ...')
    # Create a dict mapping from (corpus, piece) to year from metadata_df
    mapping = dict(zip(zip(metadata['corpus'], metadata['piece']), metadata['composed_end']))

    harmonies["piece_year"] = harmonies.apply(lambda row: mapping[(row['corpus'], row['piece'])],
                                              axis=1)  # add the year info
    final_df = harmonies.assign(corpus_year=harmonies.groupby("corpus")["piece_year"].transform(np.mean)).sort_values(
        ['corpus_year', 'piece_year']).reset_index(drop=True)
    final_df = final_df.drop(columns=['Unnamed: 0'])

    path = f'{save_path}{save_df_name}.tsv'
    print(f'Preprocess Step 7: Saving df to {path} ...')
    # final_df.to_csv(f'{path}', sep="\t", index=True)
    final_df.to_csv(f'{path}', sep="\t")

    return final_df


if __name__ == "__main__":
    # preprocess_df_Cleaning(corpus_name="distant_listening_corpus",
    #                        additional_pieces_to_exclude=[
    #                            ("frescobaldi_fiori_musicali", "12.31_Toccata_per_l'Elevatione"),
    #                            ("frescobaldi_fiori_musicali", "12.33_Canzon_quarti_toni_dopo_il_post_Comune"),
    #                            ("frescobaldi_fiori_musicali", "12.45_Toccata_per_l'Elevatione")],
    #                        save_path="../Data/prep_data/")
    #
    metadata: pd.DataFrame = load_dfs_from_corpus(corpus_name="distant_listening_corpus", df_type=["metadata"])[0]
    clean_harmonies = load_tsv_as_df(path="../Data/prep_data/cleaned_distant_listening_corpus_harmonies.tsv")
    clean_notes = load_tsv_as_df(path="../Data/prep_data/cleaned_distant_listening_corpus_notes.tsv")

    preprocess_df_AppendingNotes(metadata=metadata, harmonies=clean_harmonies, notes=clean_notes,
                                 save_path="../Data/prep_data/",
                                 save_df_name="DLC_data")
