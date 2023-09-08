
import mplfinance as fplt
import numpy as np
from pandas import wide_to_long


def plot_pivots(df, col_pivot, title='Title'):

    def pivot_pos(x, pivot):
        if x[pivot] == -1:
            return x['Low']
        elif x[pivot] == 1:
            return x['High']
        else:
            return np.nan

    def pivot_high_pos(x, pivot):
        if x[pivot] == 1:
            return x['High']
        else:
            return np.nan

    def pivot_low_pos(x, pivot):
        if x[pivot] == -1:
            return x['Low']
        else:
            return np.nan

    pivot_points = df.apply(lambda row: pivot_pos(row, col_pivot), axis=1)
    pivot_points_high = df.apply(lambda row: pivot_high_pos(row, col_pivot), axis=1)
    pivot_points_low = df.apply(lambda row: pivot_low_pos(row, col_pivot), axis=1)

    pivots = fplt.make_addplot(
        pivot_points,
        type="scatter",
        color="red",
        marker="x",
        alpha=0.7,
        markersize=50
    )

    pivots_high = fplt.make_addplot(
        pivot_points_high,
        type="scatter",
        color="green",
        marker="x",
        alpha=0.7,
        markersize=50
    )

    pivots_low = fplt.make_addplot(
        pivot_points_low,
        type="scatter",
        color="red",
        marker="x",
        alpha=0.7,
        markersize=50
    )

    padding = np.max(df['High']) - np.min(df['Low'])
    padding *= 0.1

    plot_type = 'line' if len(df) > 300 else 'ohlc_bars'  # candlestick

    fplt.plot(
        df,
        addplot=[pivots_high, pivots_low],
        type=plot_type,
        style='charles',
        title=title,
        figscale=2,
        # volume=True,
        # show_nontrading=True,
        # Set the Y-axis limits (min_y and max_y)
        ylim=(np.min(df['Low']) - padding,
              np.max(df['High']) + padding)
    )
