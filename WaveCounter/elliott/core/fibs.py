

from enum import Enum
from typing import List


class FibonacciLevel(Enum):
    FIB_0_236 = 0.236
    FIB_0_382 = 0.382
    FIB_0_5 = 0.5
    FIB_0_618 = 0.618
    FIB_0_786 = 0.786
    FIB_0_886 = 0.886
    FIB_1_0 = 1.0
    FIB_1_272 = 1.272
    FIB_1_414 = 1.414
    FIB_1_618 = 1.618
    FIB_2_0 = 2.0
    FIB_2_272 = 2.272
    FIB_2_414 = 2.414
    FIB_2_618 = 2.618
    FIB_3_618 = 3.618
    FIB_4_0 = 4
    FIB_4_236 = 4.236
    FIB_4_618 = 4.618


class Fibonacci:
    @staticmethod
    def get_retracements(pivot1: float, pivot2: float, levels: List[FibonacciLevel]):
        # Calculate the retracement levels using a dictionary comprehension
        retracement_levels = {
            level: pivot2 - (level.value * (pivot2 - pivot1))
            for level in levels
        }

        return retracement_levels

    @staticmethod
    def get_projections(pivot1: float, pivot2: float, pivot3: float, levels: List[FibonacciLevel]):
        # Calculate the projection levels using a dictionary comprehension
        projection_levels = {
            level: pivot3 + (level.value * (pivot2 - pivot1))
            for level in levels
        }

        return projection_levels

    @staticmethod
    def is_pivot_in_levels(levels: dict, pivot: float):
        items = list(levels.items())
        for i in range(1, len(items)):
            previous_level, previous_value = items[i - 1]
            current_level, current_value = items[i]

            if (current_value <= pivot and pivot <= previous_value):
                return [items[i - 1], items[i]]

        return None
