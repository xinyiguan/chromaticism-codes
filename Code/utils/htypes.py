# submodules from the harmonytypes
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, List, Generic, Optional, TypeVar

import numpy as np
import regex_spm
from pitchtypes import SpelledPitchClass, SpelledIntervalClass, asic

T = TypeVar('T')

RomanNumeral_ScaleDegree_dict = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                 "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}
intervals_in_key_dict = {'major': asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7'])),
                         'minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7'])),
                         'harmonic_minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'M7'])),

                         'ionian': asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7'])),
                         'dorian': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'M6', 'm7'])),
                         'phrygian': asic(things=np.array(['P1', 'm2', 'm3', 'P4', 'P5', 'm6', 'm7'])),
                         'lydian': asic(things=np.array(['P1', 'M2', 'M3', 'a4', 'P5', 'M6', 'M7'])),
                         'mixolydian': asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'm7'])),
                         'aeolian': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7'])),
                         'locian': asic(things=np.array(['P1', 'm2', 'm3', 'P4', 'd5', 'm6', 'm7'])),
                         }


class Regexes:
    Key_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I)
    RomanNumeral_regex = re.compile("^(?P<alterations>(b*|#*))"
                                    "(?P<numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))"
                                    "(?P<form>(%|o|\+|M|\+M))?"
                                    "(?P<figbass>(7|65|43|42|2|64|6))?$")
    RomanDegree_regex = re.compile("^(?P<alterations>(b*|#*))"
                                   "(?P<degree>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))$")
    ArabicDegree_regex = re.compile("^(?P<alterations>(b*|#*))"
                                    "(?P<degree>(\d+))$")


@dataclass
class Degree:
    number: int
    alteration: int  # sign (+/-) indicates direction (♯/♭), value indicates number of signs added

    def __str__(self):
        if self.alteration == 0:
            sign = ''
        elif self.alteration > 0:
            sign = "#" * abs(self.alteration)
        elif self.alteration < 0:
            sign = "b" * abs(self.alteration)
        else:
            raise ValueError(f'invalid {self.alteration=}')
        return f'{sign}{self.number}'

    def __add__(self, other: Degree) -> Degree:
        """
        n steps (0 steps is unison) <-- degree (1 is unison)
        |
        V
        n steps (0 steps is unison) --> degree (1 is unison)

        """
        number = ((self.number - 1) + (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    def __sub__(self, other: Degree) -> Degree:
        number = ((self.number - 1) - (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    @classmethod
    def from_string(cls, s: str) -> Degree:
        """
        Examples of arabic_degree: b7, #2, 3, 5, #5, ...
        Examples of scale degree: bV, bIII, #II, IV, vi, vii, V/V
        """

        match = regex_spm.fullmatch_in(s)
        match match:
            case Regexes.RomanDegree_regex:
                degree_number = RomanNumeral_ScaleDegree_dict.get(match['degree'])
            case Regexes.ArabicDegree_regex:
                degree_number = int(match['degree'])
            case _:
                raise ValueError(
                    f"could not match {match} with regex: {Regexes.RomanDegree_regex} or {Regexes.ArabicDegree_regex}")
        modifiers_match = match['alterations']
        alteration = SpelledPitchClass(f'C{modifiers_match}').alteration()
        instance = cls(number=degree_number, alteration=alteration)
        return instance

    @classmethod
    def from_sic(cls, sic: SpelledIntervalClass, mode=Literal["major", "minor"]) -> Degree:
        """Create a Degree object from spelled interval class"""

        if mode == 'major':
            intervals = intervals_in_key_dict['major']

        elif mode == 'minor':
            intervals = intervals_in_key_dict['minor']

        else:
            raise ValueError(f'Invalid mode {mode}')

        number = sic.degree() + 1
        alteration_degree = sic - intervals[number - 1]

        if "d" in str(alteration_degree):
            alteration = -len([c for c in str(alteration_degree) if c.isalpha()])
        elif "a" in str(alteration_degree):
            alteration = len([c for c in str(alteration_degree) if c.isalpha()])
        elif "P" in str(alteration_degree):
            alteration = 0
        else:
            raise ValueError

        instance = cls(number=number, alteration=alteration)
        return instance

    @classmethod
    def from_fifth(cls, fifth: int, mode=Literal["major", "minor"]) -> Degree:

        sic = SpelledIntervalClass(fifth)

        instance = cls.from_sic(sic=sic, mode=mode)
        return instance

    def sic(self, mode: Literal["major", "minor"]) -> SpelledIntervalClass:
        """Return the spelled interval class from the reference tonic (scale degree 1)"""

        if self.alteration > 0:
            alteration_symbol = 'a' * self.alteration
        elif self.alteration == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif self.alteration < 0:
            alteration_symbol = 'd' * abs(self.alteration)
        else:
            raise ValueError(self.alteration)

        if mode == 'major':
            intervals = intervals_in_key_dict['major']

        elif mode == 'minor':
            intervals = intervals_in_key_dict['minor']

        else:
            raise NotImplementedError(f'{mode=}')

        interval_alteration = SpelledIntervalClass(f'{alteration_symbol}1')
        interval_from_tonic = intervals[self.number - 1] + interval_alteration
        return interval_from_tonic

    def fifth(self, mode: Literal["major", "minor"]) -> int:
        result = self.sic(mode=mode).fifths()
        return result

    def roman_numeral(self) -> str:
        sd_rn_dict = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}
        rn = sd_rn_dict[self.number]
        if self.alteration == 0:
            sign = ''
        elif self.alteration > 0:
            sign = "#" * abs(self.alteration)
        elif self.alteration < 0:
            sign = "b" * abs(self.alteration)
        else:
            raise ValueError(f'invalid {self.alteration=}')
        result = f'{sign}{rn}'
        return result

    def spc(self, key: str) -> SpelledPitchClass:
        k = Key.from_string(s=key)
        result = k.find_spc_from_degree(self)
        return result


@dataclass
class Key:
    tonic: SpelledPitchClass
    mode: Literal[
        'major', 'minor', 'melodic_minor', 'harmonic_minor', 'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']

    @classmethod
    def from_string(cls, s: str) -> Key:

        if not isinstance(s, str):
            raise TypeError(f"Expected string as input, got {s}")
        key_match = Regexes.Key_regex.fullmatch(s)
        if key_match is None:
            raise ValueError(f"Could not match '{s}' with regex: '{Regexes.Key_regex.pattern}'")

        mode = 'major' if key_match['class'].isupper() else 'minor'
        tonic = SpelledPitchClass(key_match['class'].upper() + key_match['modifiers'])
        instance = cls(tonic=tonic, mode=mode)
        return instance

    def accidentals(self) -> int:
        return abs(self.relative().tonic.fifths()) if self.mode == 'minor' else abs(self.tonic.fifths())

    def relative(self) -> Key:
        if self.mode == "major":
            result = Key(tonic=self.tonic - SpelledIntervalClass("m3"), mode="minor")
        elif self.mode == "minor":
            result = Key(tonic=self.tonic + SpelledIntervalClass("m3"), mode="major")
        else:
            raise NotImplementedError
        return result

    def get_scale_members(self) -> List[SpelledPitchClass]:
        intervals = intervals_in_key_dict[self.mode]
        scale_mem = [self.tonic + intervals[i] for i in range(len(intervals))]
        return scale_mem

    def find_spc_from_degree(self, degree: Degree) -> SpelledPitchClass:
        """Minor refers to natural minor"""

        if self.mode == 'major':
            intervals = intervals_in_key_dict['major']
        elif self.mode == 'minor':
            intervals = intervals_in_key_dict['minor']
        else:
            raise NotImplementedError(f'{self.mode=}')

        if degree.alteration > 0:
            alteration_symbol = 'a' * degree.alteration
        elif degree.alteration == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif degree.alteration < 0:
            alteration_symbol = 'd' * abs(degree.alteration)
        else:
            raise ValueError(degree.alteration)

        interval_alteration = SpelledIntervalClass(f'{alteration_symbol}1')
        interval = intervals[degree.number - 1] + interval_alteration
        spc = self.tonic + interval
        return spc

    def find_degree(self, spc: SpelledPitchClass) -> Degree:
        """
        Example: in Db major scale,
                - F# will be #3
                - E will be #2
                - E# will be ##2
        """
        scale_without_accidentals = [x.letter() for x in self.get_scale_members()]
        position_in_scale = scale_without_accidentals.index(spc.letter())
        degree_num_part = position_in_scale + 1

        target_alteration = spc.alteration()
        original_scale_alteration = self.get_scale_members()[position_in_scale].alteration()
        alteration_num = target_alteration - original_scale_alteration

        if alteration_num > 0:
            alteration_symbol = '#' * alteration_num
        elif alteration_num == 0:
            alteration_symbol = ''
        elif alteration_num < 0:
            alteration_symbol = 'b' * abs(alteration_num)
        else:
            raise ValueError(spc.alteration)

        ensemble_degree_string = alteration_symbol + str(degree_num_part)

        result_degree = Degree.from_string(s=ensemble_degree_string)
        return result_degree

    @staticmethod
    def get_spc_from_fifths(k: str, fifth_step: int) -> str:
        """
        This function takes the fifth step of a pitch in key k and returns the spelled pitch class.
        Building <fifth_steps> of 5th from the tonic of the key k.
        For instance, get_spc_from_fifths(fifth_step=-1 , k="a")=D
        :param k:
        :param fifth_step: int
        :return:
        """
        tonic = Key.from_string(s=k).tonic
        if fifth_step == 0:
            ic = SpelledIntervalClass("P1")
        else:
            ic = SpelledIntervalClass("P5") * fifth_step
        result = tonic + ic
        return result

    def to_str(self) -> str:
        if self.mode == "major":
            return self.tonic.name()
        elif self.mode == "minor":
            return self.tonic.name().lower()
        else:
            raise NotImplementedError


@dataclass
class NumeralQuality:
    root: Literal[0, 1, -1]
    third: Literal[0, 1, -1]
    fifth: Literal[0, 1, -1]
    seventh: Optional[Literal[0, 1, -1, -2]]

    """
    The integers indicate the alterations from the diatonic version.
    Examples:
    <bII in C major>: {Db, F, Ab} ==> (root=-1, third=1, fifth=0, seventh=0)

    """

    @staticmethod
    def match_third_quality(diatonic_numeral: str, numeral_to_examine: str) -> int:
        match (diatonic_numeral, numeral_to_examine):
            case (x, y) if x.islower() and y.isupper():
                return 1
            case (x, y) if x.isupper() and y.islower():
                return -1
            case (x, y) if x.islower() and y.islower():
                return 0
            case (x, y) if x.isupper() and y.isupper():
                return 0
            case _:
                raise ValueError

    @staticmethod
    def match_fifth_seventh_quality(the_form: str, the_figbass: str) -> List[int]:
        """
        For seveth chord, the fifth_quality and seventh_quality will be:
        - dominant 7th  (X7):   (1, 3, 5, b7)   =>  [0, -1]
        - minor 7th     (x7):   (1, b3, 5, b7)  =>  [0, -1]
        - dim 7th       (xo7):  (1, b3, b5, bb7) => [-1, -2]
        - half-dim 7th  (x%7)   (1, b3, b5, b7) =>  [-1, -1]
        - aug 7th       (X+7):  (1, 3, #5, b7)  =>  [1, -1]
        - Major 7th     (XM7):  (1, 3, 5, 7)    =>  [0, 0]
        - minor major 7th (xM7):(1, b3, 5, 7)   =>  [0, 0]
        - aug maj 7th   (X+M7): (1, 3, #5, 7)   =>  [1, 0]
        """
        triad = ["", "6", "64"]
        tetrad = ["7", "65", "43", "42", "2"]

        match (the_form, the_figbass):
            case ("", b) if b in triad:
                return [0, None]
            case ("+", b) if b in triad:
                return [1, None]
            case ("o", b) if b in triad:
                return [-1, None]
            case ("", b) if b in tetrad:
                return [0, -1]
            case ("o", b) if b in tetrad:
                return [-1, -2]
            case ("%", b) if b in tetrad:
                return [-1, -1]
            case ("+", b) if b in tetrad:
                return [1, -1]
            case ("M", b) if b in tetrad:
                return [0, 0]
            case ("+M", b) if b in tetrad:
                return [1, 0]
            case _:
                raise ValueError

    @classmethod
    def from_numeral_parts(cls, k: Key, alteration: str, numeral: str, form: str, figbass: str) -> NumeralQuality:
        MajorDiatonicChord_Degree_dict = {1: "I", 2: "ii", 3: "iii", 4: "IV", 5: "V", 6: "vi", 7: "viio"}
        MinorDiatonicChord_Degree_dict = {1: "i", 2: "iio", 3: "III", 4: "iv", 5: "v", 6: "VI", 7: "VII"}

        rn_scale_degree = RomanNumeral_ScaleDegree_dict[numeral]
        root_str = str(alteration) + str(RomanNumeral_ScaleDegree_dict[numeral])
        root_dg = Degree.from_string(root_str)
        root_qualtiy = root_dg.alteration

        # compare with the corresponding diatonic chord:
        if (m := k.mode) == "major":
            diatonic_version_numeral = MajorDiatonicChord_Degree_dict[rn_scale_degree]
        elif m == "minor":
            diatonic_version_numeral = MinorDiatonicChord_Degree_dict[rn_scale_degree]
        else:
            raise ValueError

        third_quality = NumeralQuality.match_third_quality(diatonic_version_numeral, numeral)

        fifth_quality, seventh_quality = NumeralQuality.match_fifth_seventh_quality(the_form=form, the_figbass=figbass)

        quality = NumeralQuality(root=root_qualtiy, third=third_quality, fifth=fifth_quality, seventh=seventh_quality)

        return quality


@dataclass
class SimpleNumeral:
    key: Key
    root: SpelledPitchClass
    bass: SpelledPitchClass
    quality: NumeralQuality  # alteration with reference to the underlying key/scale of the numeral

    @classmethod
    def from_string(cls, s: str, k: Key) -> SimpleNumeral:
        """
        e.g., "V65", "bII6", viio7
        """
        if not isinstance(s, str):
            raise TypeError(f"Expected string as input, got {s}")

        FiguredBass_BassDegree_dict = {"7": "1", "65": "3", "43": "5",
                                       "42": "7", "2": "7",
                                       "64": "5",
                                       "6": "3",
                                       "": "1"}  # assume the first number x is: root + x = bass

        # match with regex
        match = Regexes.RomanNumeral_regex.match(s)
        alteration = match["alterations"] if match["alterations"] else ""
        numeral = match["numeral"]
        form = match["form"] if match["form"] else ""
        figbass = match["figbass"] if match["figbass"] else ""

        if match is None:
            raise ValueError(f"Could not match '{s}' with regex: '{Regexes.RomanNumeral_regex.pattern}'")

        root_str = str(alteration) + str(RomanNumeral_ScaleDegree_dict[numeral])
        root_dg = Degree.from_string(root_str)
        root_spc = k.find_spc_from_degree(degree=root_dg)

        root_mode = "major" if numeral.isupper() else "minor"
        key_from_root = Key(tonic=root_spc, mode=root_mode)
        bass_degree = Degree.from_string(s=FiguredBass_BassDegree_dict[figbass])
        the_bass = key_from_root.find_spc_from_degree(bass_degree)

        # deviation from the diatonic chord:
        quality = NumeralQuality.from_numeral_parts(k=k, alteration=alteration, numeral=numeral, form=form,
                                                    figbass=figbass)

        instance = cls(key=k, root=root_spc, bass=the_bass, quality=quality)
        return instance

    def numeral_string(self) -> str:

        if self.quality.root > 0:
            root_accidental = "#" * abs(self.quality.root)
        elif self.quality.root < 0:
            root_accidental = "b" * abs(self.quality.root)
        else:
            root_accidental = ""

        MajorDiatonicChord_Degree_dict = {1: "I", 2: "ii", 3: "iii", 4: "IV", 5: "V", 6: "vi", 7: "viio"}
        MinorDiatonicChord_Degree_dict = {1: "i", 2: "iio", 3: "III", 4: "iv", 5: "v", 6: "VI", 7: "VII"}

        # get the scale step (the numeral -case insensitive)
        scale_step_degree = self.key.find_degree(spc=self.root).number
        if self.key.mode == "major":
            diatonic_scale_step = MajorDiatonicChord_Degree_dict[scale_step_degree]

        elif self.key.mode == "minor":
            diatonic_scale_step = MinorDiatonicChord_Degree_dict[scale_step_degree]
        else:
            raise NotImplementedError

        match = Regexes.RomanNumeral_regex.match(diatonic_scale_step)
        diatonic_scale_step = match["numeral"]

        # get the numeral part (adjust the case - determine major/minor quality - case sensitive)
        if self.quality.third > 0:
            numeral_part = diatonic_scale_step.upper()
        elif self.quality.third < 0:
            numeral_part = diatonic_scale_step.lower()
        else:
            numeral_part = diatonic_scale_step

        # Get the inversion
        SeventhChord_FiguredBass_dict = {"1": "7", "3": "65", "5": "43",
                                         "7": "42"}
        Triad_FiguredBass_dict = {"5": "64",
                                  "3": "6",
                                  "1": ""}

        # find the bass degree counting from root
        bass_degree_from_root = str(
            Key(tonic=self.root, mode="major" if numeral_part.isupper() else "minor").find_degree(spc=self.bass).number)

        if self.quality.seventh:  # we have the seventh chord
            # inversion = SeventhChord_FiguredBass_dict[bass_degree]
            match (self.quality.fifth, self.quality.seventh):
                case (0, -1):
                    form = ""
                case (-1, -2):
                    form = "o"
                case (-1, -1):
                    form = "%"
                case (1, -1):
                    form = "+"
                case (0, 0):
                    form = "M"
                case (1, 0):
                    form = "+M"
                case _:
                    raise ValueError

        else:  # we have the triads
            match self.quality.fifth:
                case 0:
                    form = ""
                case 1:
                    form = "+"
                case -1:
                    form = "o"
                case _:
                    raise ValueError

            # inversion = Triad_FiguredBass_dict[bass_degree]
        inversion = Triad_FiguredBass_dict.get(
            bass_degree_from_root) if not self.quality.seventh else SeventhChord_FiguredBass_dict.get(
            bass_degree_from_root)

        if inversion is None:
            raise ValueError

        n_str = root_accidental + numeral_part + form + inversion
        return n_str

    def key_if_tonicized(self) -> Key:

        tonic = self.root
        mode = "major" if self.numeral_string().isupper() else "minor"
        k = Key(tonic=tonic, mode=mode)
        return k

    def parse_in_chord_tones(self) -> List[SpelledPitchClass]:
        raise NotImplementedError



@dataclass
class Chain(Generic[T]):
    head: T
    tail: Optional[Chain[T]]


@dataclass
class Numeral(Chain[SimpleNumeral]):

    @classmethod
    def from_string(cls, s: str, k: Key | str) -> Numeral:
        """
        Parse a string as a numeral.
        e.g., "V65", "bII6", "V7/V"
        """
        if not isinstance(s, str):
            raise TypeError(f"Expected string as input, got {s}")

        if isinstance(k, str):
            k = Key.from_string(s=k)

        if "/" in s:
            L_numeral_str, R_numeral_str = s.split("/", maxsplit=1)
            tail = cls.from_string(k=k, s=R_numeral_str)
            head = SimpleNumeral.from_string(k=tail.head.key_if_tonicized(), s=L_numeral_str)

        else:
            head = SimpleNumeral.from_string(k=k, s=s)
            tail = None

        instance = cls(head=head, tail=tail)
        return instance

    def numeral_string(self) -> str:

        if isinstance(self.tail, Numeral):
            n_str = self.head.numeral_string() + "/" + self.tail.numeral_string()
        else:
            n_str = self.head.numeral_string()

        return n_str

    def key_if_tonicized(self) -> Key:
        result_key = self.head.key_if_tonicized()
        return result_key

    @staticmethod
    def get_spc_from_numeral(n: str, k: str) -> str:
        key = Key.from_string(s=k)
        numeral = Numeral.from_string(s=n, k=key)
        result = numeral.key_if_tonicized().tonic.name()
        return result


if __name__ == "__main__":
    # to transform the globalkey (type=str) to a Key object
    globalkey = Key.from_string(s="a")  # suppose we have globalkey as G major

    # localkey in roman numeral (type=str)
    localkey_str = "v"  # suppose the localkey is a V (the fifth scale degree)

    # construct a Numeral object
    localkey_numeral = Numeral.from_string(s=localkey_str, k=globalkey)

    # make the numeral into a Key object:
    actual_localkey = localkey_numeral.key_if_tonicized()

    # get the str:
    actual_lk_str = actual_localkey.to_str()

    print(f'{actual_localkey=}')
    print(f'{actual_lk_str=}')
