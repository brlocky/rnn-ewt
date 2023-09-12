

from ..types import Degree, Trend, Wave
from ..core.wave_node import WaveNode
from ..core.fibs import Fibonacci, FibonacciLevel
from ..core.elliott_wave_base import ElliottWaveBase


class Wave1(ElliottWaveBase):
    def validate_pivots(self):
        if len(self.pivots) < 3:
            return None

        waves = []
        pivot1 = self.pivots[0]
        remain_pivots = self.pivots[1:].copy()
        for i in range(0, len(remain_pivots) - 1, 2):
            pivot2 = remain_pivots[i]
            pivot3 = remain_pivots[i + 1]
            levels = self.fibs.get_retracements(
                pivot1.price, pivot2.price, [
                    FibonacciLevel.FIB_0_236, FibonacciLevel.FIB_0_382,
                    FibonacciLevel.FIB_0_5, FibonacciLevel.FIB_0_618, FibonacciLevel.FIB_0_886
                ])

            result = self.fibs.is_pivot_in_levels(levels, pivot3.price)
            if result is not None:
                waves.append(WaveNode(
                    degree=Degree._SUB_MINUETTE,
                    wave=Wave._1,
                    trend=Trend.UP if pivot1.price < pivot1.price else Trend.DOWN,
                    pivots=[pivot1, pivot2, pivot3],
                    retracement=result
                ))

        print('Wave1', waves)
        return waves

    def analyze(self):
        # Implementation specific to CorrectiveWave
        pass

    def find(self):
        pass

    def validate(self):
        pass

    def add(self):
        pass

    def discard(self):
        pass
