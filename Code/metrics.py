# This script contains the functions for different measures of chromaticism and
import itertools
from typing import List, Optional, Literal

import numpy as np
from pitchtypes import SpelledPitchClass, SpelledIntervalClass
from setuptools import sic


# set chromaticity ___________________________________________________________________________________________________
# plain version, not differentiating major/minor scale mode
def _tone_to_diatonic_set_distance(tonic: Optional[SpelledPitchClass], tone: SpelledPitchClass | int) -> int:
    def _generate_diatonic_set(t: SpelledPitchClass):
        diatonic_set_5 = [t]
        current_pitch = t
        for _ in range(5):
            current_pitch = current_pitch + SpelledIntervalClass("P5")
            diatonic_set_5.append(current_pitch)
        complete_diatonic_set = [t - SpelledIntervalClass("P5")] + diatonic_set_5
        return complete_diatonic_set

    if isinstance(tone, SpelledPitchClass):
        if tonic is None:
            raise ValueError(f'Missing tonic input!')
        diatonic_set_tones = _generate_diatonic_set(t=tonic)
        if tone in diatonic_set_tones:
            d = 0
        else:
            d = min(
                abs(tone.interval_from(diatonic_set_tones[0]).fifths()),
                abs(tone.interval_from(diatonic_set_tones[-1]).fifths()))
        return d
    elif isinstance(tone, int):
        if -1 <= tone <= 5:
            d = 0
        else:
            d = min(abs(tone - 5), abs(tone + 1))
        return d
    else:
        raise TypeError


def _cumulative_distance_to_diatonic_set(tonic: Optional[SpelledPitchClass],
                                         ts: List[SpelledPitchClass] | List[int]) -> int:
    ds = []
    for t in ts:
        d = tone_to_diatonic_set_distance(tonic=tonic, tone=t)
        ds.append(d)
    total_distance = sum(ds)
    return total_distance


# set chromaticity ___________________________________________________________________________________________________
# plain version, not differentiating major/minor scale mode

def generate_diatonic_set(t: SpelledPitchClass, mode: Literal["major", "minor"]):
    ds_upper_tones = [t]
    ds_lower_tones = [t]
    current_pitch = t

    if mode == "major":
        num_upper_tones = 5
        num_lower_tones = 1
    elif mode == "minor":
        num_upper_tones = 2
        num_lower_tones = 4

    else:
        raise ValueError

    for _ in range(num_upper_tones):
        current_pitch = current_pitch + SpelledIntervalClass("P5")
        ds_upper_tones.append(current_pitch)
    for _ in range(num_lower_tones):
        current_pitch = current_pitch - SpelledIntervalClass("P5")
        ds_lower_tones.append(current_pitch)

    complete_diatonic_set = ds_lower_tones + ds_upper_tones
    return complete_diatonic_set


def tone_to_diatonic_set_distance(tonic: Optional[SpelledPitchClass],
                                  tone: SpelledPitchClass | int,
                                  diatonic_mode: Literal["major", "minor"]) -> int:
    if isinstance(tone, SpelledPitchClass):
        if tonic is None:
            raise ValueError(f'Missing tonic input!')
        diatonic_set_tones = generate_diatonic_set(t=tonic, mode=diatonic_mode)

        if tone in diatonic_set_tones:
            d = 0
        else:
            d = min(
                abs(tone.interval_from(diatonic_set_tones[0]).fifths()),
                abs(tone.interval_from(diatonic_set_tones[-1]).fifths()))
        return d

    elif isinstance(tone, int):
        if diatonic_mode == "major":
            if -1 <= tone <= 5:
                d = 0
            else:
                d = min(abs(tone - 5), abs(tone + 1))
        elif diatonic_mode == "minor":
            if -4 <= tone <= 2:
                d = 0
            else:
                d = min(abs(tone - 2), abs(tone + 4))

        else:
            raise ValueError
        return d
    else:
        raise TypeError


def cumulative_distance_to_diatonic_set(tonic: Optional[SpelledPitchClass],
                                        ts: List[SpelledPitchClass] | List[int],
                                        diatonic_mode: Literal["major", "minor"]) -> int:
    ds = []
    for t in ts:
        d = tone_to_diatonic_set_distance(tonic=tonic, tone=t, diatonic_mode=diatonic_mode)
        ds.append(d)
    total_distance = sum(ds)
    return total_distance


# diatonicity ________________________________________________________________________________________________________

def all_Ls(S: List[int]) -> List[List[int]]:
    """Given a list of integers S, we find all possible Ls (L: List[int])
    we use a sliding window with length 7 to get all possible within the range of S.
    """

    possible_starting_pos_for_L = list(range(min(S), max(S) + 1))
    Ls = [list(range(x, x + 7)) for x in possible_starting_pos_for_L]

    return Ls


def distance_from_S_to_L(S: List[int], L: List[int]) -> int:
    """
    The distance of set S to set L is defined as: the number
    D(S, L) = sum of y in {S-L} of min(d(y, max L), d(y, min L))
    :param S:
    :param L:
    :return:
    """
    result = sum([min(abs(y - min(L)), abs(y - max(L))) for y in S if y not in L])
    return result


def min_distance_from_S_to_L(S: List[int]) -> int:
    """
    min_L D(S, L): the minimal distance from a TPC set S to a diatonic set L.
    :param S:
    :return:
    """
    if S is None:
        raise ValueError('Missing data!')
    if S == []:
        return 0
    else:
        possible_Ls = all_Ls(S=S)
        result = min([distance_from_S_to_L(S=S, L=L) for L in possible_Ls])
    return result


# dissonance ____________________________________________________________________________________________________

dissonance_ranks = {
    'P5': 1,
    'P4': 1,
    'M3': 2,
    'm6': 2,
    'm3': 3,
    'M6': 3,
    'M2': 4,
    'm7': 4,
    'a4': 5,
    'd5': 5,
    'm2': 6,
    'M7': 6
}


def tpcs_to_ics(tpcs: List[int]) -> List[SpelledIntervalClass]:
    all_pairs = list(itertools.combinations(tpcs, 2))
    diffs = [np.abs(p[0] - p[1]) for p in all_pairs]
    sics = [SpelledIntervalClass.from_fifths(fifths=f) for f in diffs]
    return sics


def pcs_dissonance_rank(tpcs: List[int]) -> int:
    """
    This function takes a list of TPCs (of a chord) and returns the pairwise interval classes.
    :param tpcs: the pitch class set of the "chord" in TPC.
    :return:
    """
    sics = tpcs_to_ics(tpcs)
    rank_score = sum([dissonance_ranks[sic.name()] for sic in sics])
    return rank_score

def _dissonance_binary_score(tpc: SpelledIntervalClass) -> int:
    consonance = ["P5", "P4", "M3", "M6", "m3", "m6"]
    dissonance = ["m2", "M2", "m7", "M7", "a4", "d5"]
    if tpc.name() in consonance:
        return 0
    elif tpc.name() in dissonance:
        return 1
    else:
        raise ValueError()

def pcs_dissonance_binary(tpcs: List[int])->int:
    """
    :param pcs: the pitch class set of the "chord" in TPC
    """
    sics = tpcs_to_ics(tpcs)
    score = sum([_dissonance_binary_score(x) for x in sics])
    return score



if __name__ == "__main__":
    pcs_dissonance_binary(tpcs=[0, -3, 1])
