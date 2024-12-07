import os

import pandas as pd

from Code.utils.util import load_file_as_df


def compute_ILC_pieces_stats(df: pd.DataFrame):
    """take the piece_level_indices_by_mode.pickle"""
    major_df = df.loc[df['localkey_mode'] == "major"]
    ILC_0_major = major_df.loc[major_df['ILC'] == 0]
    num_ILC_0_major = ILC_0_major.shape[0]
    percentg_ILC_0_major = num_ILC_0_major / major_df.shape[0]
    print(f'{num_ILC_0_major=}, {percentg_ILC_0_major=}')
    print("\n")

    minor_df = df.loc[df['localkey_mode'] == "minor"]
    ILC_0_minor = minor_df.loc[minor_df['ILC'] == 0]
    num_ILC_0_minor = ILC_0_minor.shape[0]
    print(f'{num_ILC_0_minor=}')
    print(ILC_0_minor)


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/Codes/chromaticism-codes/'
    piece_idx_df = load_file_as_df(path=f"{repo}Data/prep_data/for_analysis/piece_level_indices_by_mode.pickle")
    compute_ILC_pieces_stats(piece_idx_df)
