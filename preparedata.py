import os
import shutil
from matplotlib import pyplot as plt
import mplfinance as mpf
import pandas as pd
import yfinance
from zigzag import peak_valley_pivots

from EwClassifier.utils_plot import plot_pivots


# Sample data (replace this with your own data)

# Define the path to the data folder
csv_directory = 'csv_data'

# Delete the directory and its contents if it exists
if os.path.exists(csv_directory):
    shutil.rmtree(csv_directory)

# Create the directory
os.makedirs(csv_directory)

tech_stocks_and_indexes = [
    "AAPL",    # Apple Inc.
    "MSFT",    # Microsoft Corporation
    "GOOGL",   # Alphabet Inc. (Google)
    "AMZN",    # Amazon.com Inc.
    "META",      # Meta Platforms, Inc. (Facebook)
    "TSLA",    # Tesla, Inc.
    "NVDA",    # NVIDIA Corporation
    "ADBE",    # Adobe Inc.
    "CRM",     # Salesforce.com Inc.
    "QCOM",    # Qualcomm Incorporated
    "IBM",     # International Business Machines Corporation
    "^GSPC",   # S&P 500 Index
    "^IXIC",   # NASDAQ Composite Index
    "^DJI"     # Dow Jones Industrial Average
]


for symbol in tech_stocks_and_indexes:
    try:
        data = yfinance.download(symbol, start="2022-01-01",
                                 end="2023-08-05", interval="1h")

        # Process ZIGZAG
        X = data['Close']
        pivots = peak_valley_pivots(X, 0.01, -0.01)
        zigzap = data.copy()
        zigzap = zigzap.assign(Pivot=pivots)
        csv_filename = os.path.join(
            csv_directory, f"{symbol}_stock_zizzag_data.csv")
        zigzap.to_csv(csv_filename)

        # Process ZIGZAG Image
        plot_filename = os.path.join(
            csv_directory, f"{symbol}_plot.png")
        plot_pivots(X, pivots, plot_filename)

        # Process yfinance data
        csv_filename = os.path.join(csv_directory, f"{symbol}_stock_data.csv")
        data.to_csv(csv_filename)

        # Process yfinance Image
        # Rename columns to match OHLC format
        data.rename(columns={"Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume"}, inplace=True)
        chart_filename = os.path.join(
            csv_directory, f"{symbol}_candlestick.png")
        # Plot candlestick chart using mplfinance
        mpf.plot(data, type="candle", title=f"{symbol} Candlestick Chart",
                 ylabel="Price", style="charles", savefig=chart_filename)

    except Exception as e:
        print(f"Failed to download or process data for {symbol}: {e}")
