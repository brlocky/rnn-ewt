import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


def render_chart(data, buy_signal=[], sell_signal=[]):
    df = data.copy()

    df['buy_signal'] = buy_signal
    df['sell_signal'] = sell_signal

    df.index = pd.to_datetime(df.index, utc=True)

    # Create a subplot with two rows
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.1, 0.1, 0.2, 0.2])

    # Add candlestick chart to the first row
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    ), row=1, col=1)

    # Add volume bars to the second row
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
    ), row=2, col=1)

    # Add RSI from the 'feature_rsi' column to the second row
    if 'feature_rsi' in df:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['feature_rsi'],
            name='RSI'
        ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['buy_signal'],
        name='Buy',
    ), row=4, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['sell_signal'],
        name='Sell',
    ), row=5, col=1)

    # Customize layout
    fig.update_layout(
        title="Trading Chart with RSI",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='category',  # Use category type axis
            categoryarray=df.index  # Use DataFrame's index for categories
        ),
    )

    # Show the chart
    fig.show()
