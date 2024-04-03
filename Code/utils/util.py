import ast
import logging
from fractions import Fraction as frac
from typing import List, Any, Generator, TypeVar, Tuple, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pitchtypes import SpelledIntervalClass, SpelledPitchClass
from scipy.signal import argrelextrema

from Code.utils.htypes import Key
from Code.metrics import all_Ls


def int2bool(s):
    try:
        return bool(int(s))
    except:
        return s


str2inttuple = lambda l: tuple() if l == '' else tuple(int(s) for s in l.split(', '))

str2intlist = lambda l: list() if l == '' else list(int(s) for s in l.split(', '))

CONVERTERS = {
    # 'added_tones': str2inttuple,
    # 'chord_tones': str2inttuple,
    # 'all_tones_tpc_in_C': str2inttuple,
    # 'tones_in_span_in_C': str2inttuple,
    # 'tones_in_span_in_lk': str2inttuple,
    # 'within_label': str2inttuple,
    # 'out_of_label': str2inttuple,
    'duration': frac,
    'duration_qb': frac,

    'globalkey_is_minor': int2bool,
    'localkey_is_minor': int2bool,
}

STRING = 'string'  # not str

DTYPES = {
    'corpus': STRING,
    'piece': STRING,
    'piece_year': 'Int64',
    'corpus_year': 'Float64',

    'chord': STRING,
    'numeral': STRING,
    'chord_type': STRING,

    'figbass': STRING,
    'form': STRING,
    'globalkey': STRING,
    'localkey': STRING,

    'root': 'Int64',
    'bass_note': 'Int64',

    # 'duration_qb': 'Float64',

    'midi': 'Int64',
    'tpc': 'Int64',
    'name': STRING,

    'localkey_spc': STRING,
    'localkey2C': 'Int64',
}

A = TypeVar('A')

# Saving dataframes

def save_df(df: pd.DataFrame, fname: str, file_type: Literal["pickle", "tsv", "both"], directory: str):
    if file_type == "pickle":
        df.to_pickle(f'{directory}{fname}.pickle')
    elif file_type == "tsv":
        df.to_csv(f'{directory}{fname}.tsv', sep="\t")
    elif file_type == "both":
        df.to_pickle(f'{directory}{fname}.pickle')
        df.to_csv(f'{directory}{fname}.tsv', sep="\t")
    else:
        raise ValueError()


# Loading intermediate step dataframes:
def load_file_as_df(path: str) -> pd.DataFrame:
    # determine file type:
    file_type = path.split(".")[-1]

    if file_type == "tsv":
        df = pd.read_csv(path, sep="\t", dtype=DTYPES, converters=CONVERTERS, engine='python')
        tuple_cols = ["added_tones", "chord_tones", "all_tones_tpc_in_C", "tones_in_span_in_C", "tones_in_span_in_lk",
                      "within_label", "out_of_label"]
        for x in tuple_cols:
            if x in df.columns:
                # df[x] = df[x].apply(lambda s: list(ast.literal_eval(s)))
                df[x] = df[x].apply(lambda s: list(ast.literal_eval(s)))
    elif file_type == "pickle":
        df = pd.read_pickle(path)
    else:
        raise ValueError
    return df



# Function to safely parse values
def safe_literal_eval(s):
    try:
        return list([ast.literal_eval(s)])
    except (ValueError, SyntaxError):
        return []


# jitter ____________________________________________________________________________________________________________
def rand_jitter(arr, scale: float = .01):
    stdev = scale * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# flatten list _______________________________________________________________________________________________________
def flatten(lst: List[Any]) -> Generator[Any, None, None]:
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


# Function to flatten and convert elements to integers
def flatten_to_list(element):
    if isinstance(element, (list, tuple)):
        return [item for sublist in element for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
    else:
        return []


# set logger ________________________________________________________________________________________________________

def setup_logger(name, log_file, level=logging.WARNING):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# GPR plotting related ________________________________________________________________________________________________
def find_local_extrema(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ext_max_indices = argrelextrema(data, np.greater)[0]
    ext_min_indices = argrelextrema(data, np.less)[0]
    return ext_max_indices, ext_min_indices


def annotate_local_maxima(ax: plt.Axes, x: np.ndarray, y: np.ndarray, indices: np.ndarray, data: np.ndarray,
                          offset_x: int = 0, offset_y: int = 0) -> None:
    for i, (xi, yi) in enumerate(zip(x[indices], y[indices])):
        overlap = False
        for j in range(i):
            if abs(xi - x[indices[j]]) < offset_x and abs(yi - y[indices[j]]) < offset_y:
                overlap = True
                break
        if not overlap:
            ax.annotate(f'{data[indices[i]]:.2f}', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')


# corpus composer corpus Dictionary

corpus_composer_dict = {
    'gastoldi_baletti': 'Giovanni Giacomo Gastoldi',
    'peri_euridice': 'Jacopo Peri',
    'monteverdi_madrigals': 'Claudio Monteverdi',
    'sweelinck_keyboard': 'Jan Pieterszoon Sweelinck',
    'frescobaldi_fiori_musicali': 'Girolamo Frescobaldi',
    'kleine_geistliche_konzerte': 'Heinrich Schütz',
    'corelli': 'Arcangelo Corelli',
    'couperin_clavecin': 'François Couperin',
    'handel_keyboard': 'George Frideric Handel',
    'bach_solo': 'J.S. Bach',
    'bach_en_fr_suites': 'J.S. Bach',
    'couperin_concerts': ' François Couperin',
    'pergolesi_stabat_mater': 'Giovanni Battista Pergolesi',
    'scarlatti_sonatas': 'Domenico Scarlatti',
    'wf_bach_sonatas': 'W.F. Bach',
    'jc_bach_sonatas': 'J.C. Bach',
    'mozart_piano_sonatas': 'Wolfgang Amadeus Mozart',
    'pleyel_quartets': 'Ignaz Pleyel',
    'beethoven_piano_sonatas': 'Ludwig van Beethoven',
    'kozeluh_sonatas': 'Leopold Koželuh',
    'ABC': 'Ludwig van Beethoven',
    'schubert_dances': 'Franz Schubert',
    'schubert_winterreise': 'Franz Schubert',
    'mendelssohn_quartets': 'Felix Mendelssohn',
    'chopin_mazurkas': 'Frédéric Chopin',
    'schumann_kinderszenen': 'Robert Schumann',
    'schumann_liederkreis': 'Robert Schumann',
    'c_schumann_lieder': 'Clara Schumann',
    'liszt_pelerinage': 'Franz Liszt',
    'wagner_overtures': 'Richard Wagner',
    'tchaikovsky_seasons': 'Pyotr Tchaikovsky',
    'dvorak_silhouettes': 'Antonín Dvořák',
    'grieg_lyric_pieces': 'Edvard Grieg',
    'ravel_piano': 'Maurice Ravel',
    'mahler_kindertotenlieder': 'Gustav Mahler',
    'debussy_suite_bergamasque': 'Claude Debussy',
    'bartok_bagatelles': 'Béla Bartók',
    'medtner_tales': 'Nikolai Medtner',
    'poulenc_mouvements_perpetuels': 'Francis Poulenc',
    'rachmaninoff_piano': 'Sergei Rachmaninoff',
    'schulhoff_suite_dansante_en_jazz': 'Erwin Schulhoff'
}

corpus_collection_dict = {
    'gastoldi_baletti': "Balletti",
    'peri_euridice': "Euridice Opera",
    'monteverdi_madrigals': "Madrigal",
    'sweelinck_keyboard': 'Fantasia Cromatica',
    'frescobaldi_fiori_musicali': 'Fiori Musicali',
    'kleine_geistliche_konzerte': 'Kleinen geistlichen Konzerte',
    'corelli': 'Trio Sonatas',
    'couperin_clavecin': "L´Art de Toucher le Clavecin",
    'handel_keyboard': "Air and 5 variations 'The Harmonious Blacksmith'",
    'bach_solo': 'Solo Pieces',
    'bach_en_fr_suites': 'Suites',
    'couperin_concerts': "Concerts Royaux, Les Goûts-réunis",
    'pergolesi_stabat_mater': 'Stabat Mater',
    'scarlatti_sonatas': 'Harpsichord Sonatas',
    'wf_bach_sonatas': 'Keyboard Sonatas',
    'jc_bach_sonatas': 'Sonatas',
    'mozart_piano_sonatas': 'Piano Sonatas',
    'pleyel_quartets': 'String Quartets',
    'beethoven_piano_sonatas': 'Piano Sonatas',
    'kozeluh_sonatas': 'Sonatas',
    'ABC': 'String Quartets',
    'schubert_dances': 'Piano Dances',
    'schubert_winterreise': 'Winterreise',
    'mendelssohn_quartets': 'String Quartets',
    'chopin_mazurkas': 'Mazurkas',
    'schumann_kinderszenen': 'Kinderszenen',
    'schumann_liederkreis': 'Liederkreis',
    'c_schumann_lieder': 'Lieder',
    'liszt_pelerinage': 'Années de Pèlerinage',
    'wagner_overtures': 'Overtures',
    'tchaikovsky_seasons': 'The Seasons',
    'dvorak_silhouettes': 'Silhouettes',
    'grieg_lyric_pieces': 'Lyric Pieces',
    'ravel_piano': 'Piano',
    'mahler_kindertotenlieder': 'Kindertotenlieder',
    'debussy_suite_bergamasque': 'Suite Bergamasque',
    'bartok_bagatelles': 'Bagatelles',
    'medtner_tales': 'Tales',
    'poulenc_mouvements_perpetuels': 'Mouvements Perpétuels',
    'rachmaninoff_piano': 'Piano',
    'schulhoff_suite_dansante_en_jazz': 'Suite dansante en jazz'
}


# computing stats _____________________________________________________________________________________________________

def compute_fifths_range_max_min_percentage():
    df = pd.read_csv("../old_data/piecelevel_chromaticities.tsv", sep="\t")
    diatonic_roots = df["root_fifths_range"]


def fifths_range_summary():
    df = pd.read_csv("../old_data/piecelevel_chromaticities.tsv", sep="\t")

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns

    num_of_piece_with_all_diatonic_root = ((df['root_fifths_range'] >= 0) & (df['root_fifths_range'] <= 6)).sum()
    perc_of_piece_with_all_diatonic_root = num_of_piece_with_all_diatonic_root / df.shape[0]

    num_of_piece_with_all_diatonic_ct = ((df['ct_fifths_range'] >= 0) & (df['ct_fifths_range'] <= 6)).sum()
    perc_of_piece_with_all_diatonic_ct = num_of_piece_with_all_diatonic_ct / df.shape[0]

    num_of_piece_with_all_diatonic_pc = ((df['pc_fifths_range'] >= 0) & (df['pc_fifths_range'] <= 6)).sum()
    perc_of_piece_with_all_diatonic_pc = num_of_piece_with_all_diatonic_pc / df.shape[0]

    result_df = pd.DataFrame({
        'Num. of pieces with all diatonic root': [num_of_piece_with_all_diatonic_root],
        'Perc. of pieces with all diatonic root': [perc_of_piece_with_all_diatonic_root],

        'Num. of pieces with all diatonic ct': [num_of_piece_with_all_diatonic_ct],
        'Perc. of pieces with all diatonic ct': [perc_of_piece_with_all_diatonic_ct],

        'Num. of pieces with all diatonic pc': [num_of_piece_with_all_diatonic_pc],
        'Perc. of pieces with all diatonic pc': [perc_of_piece_with_all_diatonic_pc],

    })

    result_df = result_df.to_dict()
    # result_df = result_df.to_latex(index=False, escape=False)

    print(f'{result_df}')


def get_corpus_summary_table(metadata_path: str = "../data/all_subcorpora/all_subcorpora.metadata.tsv",
                             result_corpus_path: str = "../data/piecelevel_chromaticities.tsv"):
    piece_df = pd.read_csv(result_corpus_path, sep="\t")
    metadata_df = pd.read_csv(metadata_path, sep="\t")

    # Create a dictionary with a default value (e.g., 'Not Found') for unmatched keys
    # corpus_composer_dict = dict(zip(corpus_df['corpus'], metadata_df.set_index('corpus')['composer'].fillna('Not Found')))

    corpus_composer_dict = {
        'gastoldi_baletti': 'Giovanni Giacomo Gastoldi',
        'peri_euridice': 'Jacopo Peri',
        'monteverdi_madrigals': 'Claudio Monteverdi',
        'sweelinck_keyboard': 'Jan Pieterszoon Sweelinck',
        'frescobaldi_fiori_musicali': 'Girolamo Frescobaldi',
        'kleine_geistliche_konzerte': 'Heinrich Schütz',
        'corelli': 'Arcangelo Corelli',
        'couperin_clavecin': 'François Couperin',
        'handel_keyboard': 'George Frideric Handel',
        'bach_solo': 'J.S. Bach',
        'bach_en_fr_suites': 'J.S. Bach',
        'couperin_concerts': ' François Couperin',
        'pergolesi_stabat_mater': 'Giovanni Battista Pergolesi',
        'scarlatti_sonatas': 'Domenico Scarlatti',
        'wf_bach_sonatas': 'W.F. Bach',
        'jc_bach_sonatas': 'J.C. Bach',
        'mozart_piano_sonatas': 'Wolfgang Amadeus Mozart',
        'pleyel_quartets': 'Ignaz Pleyel',
        'beethoven_piano_sonatas': 'Ludwig van Beethoven',
        'kozeluh_sonatas': 'Leopold Koželuh',
        'ABC': 'Ludwig van Beethoven',
        'schubert_dances': 'Franz Schubert',
        'schubert_winterreise': 'Franz Schubert',
        'mendelssohn_quartets': 'Felix Mendelssohn',
        'chopin_mazurkas': 'Frédéric Chopin',
        'schumann_kinderszenen': 'Robert Schumann',
        'schumann_liederkreis': 'Robert Schumann',
        'c_schumann_lieder': 'Clara Schumann',
        'liszt_pelerinage': 'Franz Liszt',
        'wagner_overtures': 'Richard Wagner',
        'tchaikovsky_seasons': 'Pyotr Tchaikovsky',
        'dvorak_silhouettes': 'Antonín Dvořák',
        'grieg_lyric_pieces': 'Edvard Grieg',
        'ravel_piano': 'Maurice Ravel',
        'mahler_kindertotenlieder': 'Gustav Mahler',
        'debussy_suite_bergamasque': 'Claude Debussy',
        'bartok_bagatelles': 'Béla Bartók',
        'medtner_tales': 'Nikolai Medtner',
        'poulenc_mouvements_perpetuels': 'Francis Poulenc',
        'rachmaninoff_piano': 'Sergei Rachmaninoff',
        'schulhoff_suite_dansante_en_jazz': 'Erwin Schulhoff'
    }

    corpus_collection_dict = {
        'gastoldi_baletti': "Balletti",
        'peri_euridice': "Euridice Opera",
        'monteverdi_madrigals': "Madrigal",
        'sweelinck_keyboard': 'Fantasia Cromatica',
        'frescobaldi_fiori_musicali': 'Fiori Musicali',
        'kleine_geistliche_konzerte': 'Kleinen geistlichen Konzerte',
        'corelli': 'Trio Sonatas',
        'couperin_clavecin': "L´Art de Toucher le Clavecin",
        'handel_keyboard': "Air and 5 variations 'The Harmonious Blacksmith'",
        'bach_solo': 'Solo Pieces',
        'bach_en_fr_suites': 'Suites',
        'couperin_concerts': "Concerts Royaux, Les Goûts-réunis",
        'pergolesi_stabat_mater': 'Stabat Mater',
        'scarlatti_sonatas': 'Harpsichord Sonatas',
        'wf_bach_sonatas': 'Keyboard Sonatas',
        'jc_bach_sonatas': 'Sonatas',
        'mozart_piano_sonatas': 'Piano Sonatas',
        'pleyel_quartets': 'String Quartets',
        'beethoven_piano_sonatas': 'Piano Sonatas',
        'kozeluh_sonatas': 'Sonatas',
        'ABC': 'String Quartets',
        'schubert_dances': 'Piano Dances',
        'schubert_winterreise': 'Winterreise',
        'mendelssohn_quartets': 'String Quartets',
        'chopin_mazurkas': 'Mazurkas',
        'schumann_kinderszenen': 'Kinderszenen',
        'schumann_liederkreis': 'Liederkreis',
        'c_schumann_lieder': 'Lieder',
        'liszt_pelerinage': 'Années de Pèlerinage',
        'wagner_overtures': 'Overtures',
        'tchaikovsky_seasons': 'The Seasons',
        'dvorak_silhouettes': 'Silhouettes',
        'grieg_lyric_pieces': 'Lyric Pieces',
        'ravel_piano': 'Piano',
        'mahler_kindertotenlieder': 'Kindertotenlieder',
        'debussy_suite_bergamasque': 'Suite Bergamasque',
        'bartok_bagatelles': 'Bagatelles',
        'medtner_tales': 'Tales',
        'poulenc_mouvements_perpetuels': 'Mouvements Perpétuels',
        'rachmaninoff_piano': 'Piano',
        'schulhoff_suite_dansante_en_jazz': 'Suite dansante en jazz'
    }

    piece_num = piece_df.groupby('corpus').agg(
        Piece_Number=('piece', 'count')
    ).reset_index()

    result_df = pd.DataFrame.from_dict(corpus_composer_dict, orient='index').reset_index()
    result_df["Collection"] = result_df['index'].map(corpus_collection_dict)
    result_df = result_df.rename(columns={'index': 'corpus', '0': 'Composer'})
    result_df = pd.merge(result_df, piece_num, on='corpus', how='left')

    result_df = result_df.drop(columns=["corpus"])
    latex_table = result_df.to_latex()
    print(latex_table)


def corpus_summary_stats():
    raise NotImplementedError


if __name__ == "__main__":
    path = "Data/prep_data/cleaned_distant_listening_corpus_harmonies.tsv"
    file_type=path.split(".")[-1]
    print(f'{file_type=}')


