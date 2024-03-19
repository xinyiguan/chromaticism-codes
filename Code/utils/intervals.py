from dataclasses import dataclass
from typing import Self

from pitchtypes import SpelledPitch, SpelledPitchClass, EnharmonicIntervalClass


# post tonal set theory

@dataclass
class IntervalClass:
    # def: the smallest number of semitones possible between two pcs
    # range: 0 - 6
    integer: int

    @classmethod
    def from_pcs(cls, p1: SpelledPitch | SpelledPitchClass, p2: SpelledPitch | SpelledPitchClass) -> Self:
        integer = ...

        instance = cls(integer=integer)
        return instance


if __name__ == "__main__":
    ic = EnharmonicIntervalClass()
