# This script contains the functions for different measures of chromaticism and
import itertools
from typing import List, Optional, Literal

import numpy as np
from pitchtypes import SpelledPitchClass, SpelledIntervalClass, EnharmonicIntervalClass


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

    elif isinstance(tone, int | np.int64):
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
        raise TypeError(f'tone is {tone}. Type of tone is {type(tone)}')


def cumulative_distance_to_diatonic_set(tonic: Optional[SpelledPitchClass],
                                        ts: Optional[List[SpelledPitchClass] | List[int]],
                                        diatonic_mode: Literal["major", "minor"]) -> int:
    ds = []
    if ts:
        for t in ts:
            d = tone_to_diatonic_set_distance(tonic=tonic, tone=t, diatonic_mode=diatonic_mode)
            ds.append(d)
        total_distance = sum(ds)
    else:
        total_distance = 0
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

# enharmonic interval class rank
dissonance_ic_rank = {
    0: 0,
    1: 6,
    2: 4,
    3: 3,
    4: 2,
    5: 1,
    6: 5,
}


def _eic2ic(eic: EnharmonicIntervalClass) -> int:
    eic_val = eic.value
    assert 0 <= eic_val < 12
    return min(eic_val, (-eic_val) % 12)


def tpcs_to_ics(tpcs: Optional[List[int]]) -> List[int]:
    """
    This function takes a list of TPCs (of a chord) and returns the pairwise interval classes.
    tonal pitch class set to interval class set (unordered pitch interval class)
    """

    if tpcs:
        all_pairs = itertools.combinations(tpcs, 2)
        diffs = [min(np.abs(p[0] - p[1]), np.abs(p[1] - p[0])) for p in all_pairs]
        eics = [SpelledIntervalClass.from_fifths(fifths=f).convert_to(EnharmonicIntervalClass) for f in diffs]
        ics = [_eic2ic(x) for x in eics]
        return ics
    else:
        return []


## normalized by num of ics
# def dissonance_score(ics: List[int]) -> float:
#     weights = [1.0, 0.6, 0.4, 0.2, 0.0, 0.8]
#     pmf = np.array([ics.count(i + 1) for i in range(6)]) / len(ics)
#     res = np.nansum([pmf[i] * weights[i] for i in range(6)])
#     res = float(res)
#     return res


# sum version
def dissonance_score(ics: List[int]) -> float:
    weights = [1.0, 0.6, 0.4, 0.2, 0.0, 0.8]
    pmf = np.array([ics.count(i + 1) for i in range(6)])
    res = np.nansum([pmf[i] * weights[i] for i in range(6)])
    res = round(res, 5)
    return res


# ## log(sum+1)
# def dissonance_score(ics: List[int]) -> float:
#     """
#     Calculate dissonance of `x`, a list of interval classes, given weights for classes 1--6
#     """
#     weights = [1.0, 0.6, 0.4, 0.2, 0.0, 0.8]
#     pmf = np.array([ics.count(i + 1) for i in range(6)])
#     w_sum = round(np.nansum([pmf[i] * weights[i] for i in range(6)]), 5)
#     res = round(np.log(w_sum + 1), 5)
#     return res


# def pcs_to_dissonance_score(tpcs: List[int]) -> float:
#     ics = tpcs_to_ics(tpcs)
#     diss = dissonance_score(ics)
#     return diss

# normalize by num of chord tones
def pcs_to_dissonance_score(tpcs: List[int]) -> float:
    ics = tpcs_to_ics(tpcs)
    diss = round(dissonance_score(ics) / len(tpcs), 4)
    return diss


if __name__ == "__main__":
    # print(f'{dissonance_score(ics=[3, 4, 3, 5, 6,1])}')

    major_triad = [0, 4, 1]
    # sum-version=0.6;
    # log(sum+1)-version=0.47 ;
    # normed-by-CT-version=0.2:
    # print(f'{tpcs_to_ics(major_triad)=}')
    print(f'{pcs_to_dissonance_score(major_triad)=}')

    dom_sev = [0, 4, 1, -2]
    # sum-version=2.4;
    # log(sum+1)-version=1.22 ;
    # normed-by-CT-version=0.6:
    # print(f'{tpcs_to_ics(dom_sev)=}')
    print(f'{pcs_to_dissonance_score(dom_sev)=}')

    dim_triad = [0, -3, -6]
    # sum-version=1.6;
    # log(sum+1)-version=0.955 ;
    # normed-by-CT-version=0.533:
    # print(f'{tpcs_to_ics(dim_triad)=}')
    print(f'{pcs_to_dissonance_score(dim_triad)=}')

    dim_sev = [0, -3, -6, -9]
    # sum-version=3.2;
    # log(sum+1)-version=1.435 ;
    # normed-by-CT-version=0.8:
    # print(f'{tpcs_to_ics(dim_sev)=}')
    print(f'{pcs_to_dissonance_score(dim_sev)=}')
