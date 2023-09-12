
from enum import Enum


class Trend(Enum):
    UP = 1
    DOWN = -1


class PivotType(Enum):
    HIGH = 1
    LOW = -1


class Wave(Enum):
    _1 = 1
    _2 = 2
    _3 = 3
    _4 = 4
    _5 = 5
    _A = 6
    _B = 7
    _C = 8


class Degree(Enum):
    _SUB_MINUETTE = 1
    _MINUETTE = 2
    _MINUTE = 3
    _MINOR = 4
    _INTERMEDIATE = 5
    _PRIMARY = 6
    _CYCLE = 7
    _SUPERCYCLE = 8
    _GRAND_SUPERCYCLE = 9


class CANDLESTICKBAR(Enum):
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    PIVOT = 'Pivot'
