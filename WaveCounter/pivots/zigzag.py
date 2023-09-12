
import numpy as np

PEAK, VALLEY = 1, -1


class ZigZag(object):

    def __init__(self, df) -> None:

        # Convert prices to arrays for faster access
        self.close = df['Close'].to_numpy()
        self.high = df['High'].to_numpy()
        self.low = df['Low'].to_numpy()
        self.len = len(self.close)

        # Initialize Pivots array with length of Close
        self.pivots = [0] * self.len

        # Identify initial Pivot
        initial_pivot = self._identify_pivot_from_next(
            0,
            self.high[0],
            self.low[0]
        )

        # Update initial Pivot
        self.pivots[0] = initial_pivot

        # Initialize last pivot control variables
        self.last_pivot_type = initial_pivot
        self.last_pivot_index = 0

    # Look into the future for first High or Low to identify given pivot time and range
    def _identify_pivot_from_next(self, index, high, low):
        for t in range(index, self.len):
            if self.high[t] > high and self.low[t] > low:
                return VALLEY

            if self.low[t] < low and self.high[t] < high:
                return PEAK

        raise Exception('Could not identify Pivot')

    def update_last_pivot(self, index, new_pivot):
        if self.last_pivot_type != new_pivot:
            self.pivots[self.last_pivot_index] = self.last_pivot_type

        self.last_pivot_type = new_pivot
        self.last_pivot_index = index

    def get_zigzag(self):

        # Initialize variables to control zigzag generation
        last_high = self.high[0]
        last_low = self.low[0]

        for t in range(1, self.len):
            current_high = self.high[t]
            current_low = self.low[t]

            # Inside candle
            if current_high < last_high and current_low > last_low:
                self.update_last_pivot(t, -self.last_pivot_type)

            # Outside Candel
            elif current_high >= last_high and current_low <= last_low:
                new_pivot = self._identify_pivot_from_next(
                    t,
                    current_high,
                    current_low
                )

                self.update_last_pivot(t, new_pivot)
            else:
                if current_high > last_high:
                    self.update_last_pivot(t, PEAK)

                if current_low < last_low:
                    self.update_last_pivot(t, VALLEY)

            last_high = current_high
            last_low = current_low

        return self.pivots
