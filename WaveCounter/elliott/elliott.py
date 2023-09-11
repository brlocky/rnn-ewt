
from enum import Enum
from typing import List

class Trend(Enum):
    UP = 1
    DOWN = -1

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

class Pivot:
    def __init__(self, x:int, y:float):
        self.time = x
        self.price = y

class WaveNode:
    def __init__(self, degree:Degree, wave_from:Wave, wave_to:Wave, trend:Trend, pivots:List[Pivot]):
        self.degree = degree
        self.wave_from = wave_from
        self.wave_to = wave_to
        self.trend = trend
        self.pivots = pivots
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self, level=0):
        ret = "\t" * level + f"Degree: {self.degree}, Wave From: {self.wave_from}, Wave To: {self.wave_to}, Trend: {self.trend}\n"
        for pivot in self.pivots:
            ret += "\t" * (level + 1) + f"Pivot: {pivot.time}, {pivot.price}\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

class Elliott(object):

    def __init__(self, df, pivots, start=0, end=0) -> None:
        print(f"Elliott Start size{len(df)} Pivots{len(pivots)}")

        self.current_wave = Wave._1
        self.start = start
        self.end = end

        assert(self.start == 0 or (len(df) < self.start))
        assert(self.end == 0 or (len(df) > self.end and self.end > self.start))
    
    def _do_work(self):
        
        # Prepare tree structure
        