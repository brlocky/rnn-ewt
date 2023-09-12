from typing import List

from ..types import CANDLESTICKBAR, Degree, PivotType, Wave, Trend


class ChartData:
    def __init__(self, df):
        self.data = {
            CANDLESTICKBAR.OPEN: df[CANDLESTICKBAR.OPEN.value].tolist(),
            CANDLESTICKBAR.HIGH: df[CANDLESTICKBAR.HIGH.value].tolist(),
            CANDLESTICKBAR.LOW: df[CANDLESTICKBAR.LOW.value].tolist(),
            CANDLESTICKBAR.CLOSE: df[CANDLESTICKBAR.CLOSE.value].tolist(),
            CANDLESTICKBAR.PIVOT: df[CANDLESTICKBAR.PIVOT.value].tolist()
        }

    @property
    def size(self) -> int:
        return len(self.data[CANDLESTICKBAR.OPEN])

    @property
    def opens(self):
        return self.data[CANDLESTICKBAR.OPEN]

    @property
    def highs(self):
        return self.data[CANDLESTICKBAR.HIGH]

    @property
    def lows(self):
        return self.data[CANDLESTICKBAR.LOW]

    @property
    def closes(self):
        return self.data[CANDLESTICKBAR.CLOSE]

    @property
    def pivots(self):
        return self.data[CANDLESTICKBAR.PIVOT]


class Pivot:
    def __init__(self, x: int, y: float, type: PivotType):
        self.index = x
        self.price = y
        self.type = type


class WaveNode:
    def __init__(self, degree: Degree, wave: Wave, trend: Trend, pivots: List[Pivot], retracement):
        self.trend = trend
        self.degree = degree
        self.wave = wave
        self.pivots = pivots
        self.retracement = retracement
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self, level=0):
        ret = "\t" * level + \
            f"Degree: {self.degree}, Wave From: {self.wave_from}, Wave To: {self.wave_to}, Trend: {self.trend}\n"
        for pivot in self.pivots:
            ret += "\t" * (level + 1) + f"Pivot: {pivot.index}, {pivot.price}\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret
