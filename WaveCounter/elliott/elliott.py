
from math import inf
from .core.wave_node import ChartData
from .core.wave_search import WaveSearch
from .types import Degree, Trend, Wave


class Elliott(object):

    def __init__(self, df, start=0, end=0) -> None:
        print(f"Elliott Start size{len(df)}")

        assert (start == 0 or (len(df) < start))
        assert (start == 0 and end == 0 or (len(df) > end and end > start + 20))

        self.start = start
        self.end = end

        self.data = ChartData(df)

        self.waves = []

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

    def get_waves(self):

        # Identify current Wave Degree
        self.current_wave = Wave._1

        search = WaveSearch(self.data, Degree._SUB_MINUETTE, 0, len(self.data.opens))
        return search.run()
