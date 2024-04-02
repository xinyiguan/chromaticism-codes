from dataclasses import dataclass
from typing import Self, List

from pitchtypes import SpelledPitch, SpelledPitchClass, EnharmonicIntervalClass, EnharmonicPitchClass, \
    SpelledIntervalClass


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


@dataclass
class PitchClassInteger:
    integer: int

    @classmethod
    def from_pc_content(cls, pc_content: str) -> Self:
        integer = EnharmonicPitchClass(pc_content).name(as_int=True)
        instance = cls(integer=int(integer))

        return instance


@dataclass
class UnorderedPitchClassInterval:
    integer: int

    @classmethod
    def from_pc_content(cls, pcc1: PitchClassInteger, pcc2: PitchClassInteger) -> Self:
        p1 = pcc1.integer
        p2 = pcc2.integer

        integer = min((p1 - p2) % 12, (p2 - p1) % 12)
        instance = cls(integer=integer)
        return instance


def to_upci(pci: int) -> int:
    assert 0 <= pci < 12
    return min(pci, (-pci) % 12)


if __name__ == "__main__":
    tpc1 = -8
    tpc2 = 4
    # sp1 = SpelledPitchClass.from_fifths(tpc1).name()
    # sp2 = SpelledPitchClass.from_fifths(tpc2).name()
    # print(f'{sp1=}, {sp2=}')
    #
    # upci = UnorderedPitchClassInterval.from_pc_content(pcc1=PitchClassInteger.from_pc_content(pc_content=sp1),
    #                                                    pcc2=PitchClassInteger.from_pc_content(pc_content=sp2))
    #
    # print(f'{upci.integer=}')

    ic = SpelledIntervalClass.from_fifths(10).convert_to(EnharmonicIntervalClass).value
    print(f'{to_upci(ic)=}')
    a: EnharmonicIntervalClass
