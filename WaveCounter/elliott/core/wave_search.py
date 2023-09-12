
import pandas as pd
from typing import List

from sqlalchemy import ExceptionContext

from .wave_validator import WaveValidator

from .wave_node import ChartData, Pivot, WaveNode

from ..types import Degree, PivotType, Wave, Trend


# Identify possible wave counts for bullish and bearish approachs
# 1 - Identify current Wave Degree
# 1.1 - Information about degree and current wave
# 2 - Find next Wave
# 3 - loop with information about current degree and wave count
# 4 - Stop
# 4.1 - Defined Const for max loops
# 4.2 - Got to the end of the Wave Count
# 4.3 - Got to the end of the Wave Degree
# 5 - Print Waves
# 5.1 - Identified
# 5.2 - Projections for bullish and bearish

class WaveSearch:
    def __init__(self, chat_data: ChartData, degree: Degree, start_index: int, end_index: int):
        self.chat_data = chat_data
        self.degree = degree
        self.start_index = start_index
        self.end_index = end_index

    # Find the start pivot from given pivots
    # Search for 1st UP or DOWN pivots
    def find_first_pivot(self, pivots):
        # Search for the first non-zero pivot
        for index, pivot in enumerate(pivots):
            if pivot != 0:
                return index

        print("Could not find either UP or DOWN pivots in the pivots list")
        return None

    def run(self):
        index = self.find_first_pivot(self.chat_data.pivots)
        if index is not None:
            connections = self.generate_wave_connections(index)
            print('Got wave Connections ', connections)
            return connections
        else:
            raise Exception("No pivots found in the list")

        return 1

    # Get Trend from Index

    def get_next_pivot(self, index: int):
        current_index = index
        while current_index < self.chat_data.size:
            pivot = self.chat_data.pivots[current_index]
            if pivot != 0:
                if pivot == 1:
                    return Pivot(current_index, self.chat_data.highs[current_index], PivotType.HIGH)
                if pivot == -1:
                    return Pivot(current_index, self.chat_data.lows[current_index], PivotType.LOW)

            current_index += 1
        return None

    def generate_wave_connections(self, start_index: int):

        current_pivot_index = start_index
        pivots: List[Pivot] = []
        for i in range(10):
            current_pivot = self.get_next_pivot(current_pivot_index)
            if current_pivot is None:
                raise Exception('Could not find initial pivot')

            pivots.append(current_pivot)
            current_pivot_index = current_pivot.index + 1

        validation = WaveValidator(pivots)
        waves = validation.validate()
        print('running')
        return waves
