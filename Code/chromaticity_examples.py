import ast

import pandas as pd
from pitchtypes import SpelledPitchClass

from Code.utils.htypes import Key, Numeral
from Code.metrics import tone_to_diatonic_set_distance, cumulative_distance_to_diatonic_set
from Code.utils.util import safe_literal_eval, flatten_to_list


def CI_for_example_harmonies(tsv_path: str = "data/old_examples/roman_numerals_expanded.tsv"):
    cols2use = ["label", "globalkey", "localkey", "chord", "chord_tones", "added_tones", "root", "bass_note"]
    df = pd.read_csv(tsv_path, sep="\t", usecols=cols2use)

    def find_lk_spc(row):
        """get the local key tonic (in roman numeral) in spelled pitch class"""
        return Numeral.from_string(s=row["localkey"], k=Key.from_string(s=row["globalkey"])).key_if_tonicized().tonic

    def lk2c_dist(row):
        """get the fifths distance from the local key tonic to C"""
        return row["localkey_spc"].interval_to(SpelledPitchClass("C")).fifths()

    def determine_mode(row):
        return "minor" if row["localkey"].islower() else "major"

    df["lk_mode"] = df.apply(determine_mode, axis=1)

    df["localkey_spc"] = df.apply(find_lk_spc, axis=1)

    df["lk2C"] = df.apply(lk2c_dist, axis=1)

    df["added_tones"] = df["added_tones"].apply(lambda s: safe_literal_eval(s)).apply(flatten_to_list)
    df["chord_tones"] = df["chord_tones"].apply(lambda s: list(ast.literal_eval(s)))

    df["ct"] = df.apply(lambda row: [x for x in row["chord_tones"] + row["added_tones"] if x != row["root"]], axis=1)

    df["r_chromaticity"] = df.apply(lambda row: tone_to_diatonic_set_distance(tone=int(row["root"]),
                                                                              tonic=None,
                                                                              diatonic_mode=row["lk_mode"]), axis=1)

    # the cumulative distance of the chord tones to the local key scale set
    df["ct_chromaticity"] = df.apply(
        lambda row: cumulative_distance_to_diatonic_set(tonic=None, ts=row["ct"], diatonic_mode=row["lk_mode"]), axis=1)

    sub_df = df[["localkey", "chord", "root", "r_chromaticity", "ct", "ct_chromaticity"]]

    # Find the index where "localkey" is "i"
    index_minor = sub_df[sub_df['localkey'] == 'i'].index[0]

    sub_df_major = sub_df.iloc[:index_minor].reset_index(drop=True).set_index('chord').drop(
        columns='localkey')
    sub_df_minor = sub_df.iloc[index_minor:].reset_index(drop=True).set_index('chord').drop(
        columns='localkey')

    major_latex = sub_df_major.copy()
    major_latex.index = [f'${col}$' for col in major_latex.index]

    minor_latex = sub_df_minor.copy()
    minor_latex.index = [f'${col}$' for col in minor_latex.index]

    major_latex_table = major_latex.to_latex().replace("#", "\#").replace("%", "\%").replace("r_chromaticity", "RC").replace("ct_chromaticity", "CTC")
    minor_latex_table = minor_latex.to_latex().replace("#", "\#").replace("%", "\%").replace("r_chromaticity", "RC").replace("ct_chromaticity", "CTC")

    print(major_latex_table)
    print("____________")
    print(minor_latex_table)

    sub_df_major.to_csv("data/old_examples/major_example_CIs.tsv", sep="\t")
    sub_df_minor.to_csv("data/old_examples/minor_example_CIs.tsv", sep="\t")


if __name__ == "__main__":
    CI_for_example_harmonies()
