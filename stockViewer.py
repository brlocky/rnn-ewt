import os
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

from alpha_vantage.timeseries import TimeSeries
import requests
import yfinance as yf
from polygon import RESTClient


API_KEY = "SpQWN6pexWDTfYxepNtwonnhdxvIv24M"  # Replace with your Polygon API key

csv_directory = 'csv_data_5m'
# Create the directory
# os.makedirs(csv_directory)


class StockDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Data Viewer")
        self.client = RESTClient(api_key=API_KEY)

        # Set default values
        self.default_ticker = "AAPL"
        self.default_timeframe = "5min"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        self.recording = False
        self.recorded_candles = []

        # Create widgets
        self.label_ticker = tk.Label(root, text="Enter Stock Ticker:")
        self.entry_ticker = tk.Entry(root)
        self.entry_ticker.insert(0, self.default_ticker)
        self.label_timeframe = tk.Label(root, text="Select Timeframe:")
        self.var_timeframe = tk.StringVar()
        self.var_timeframe.set(self.default_timeframe)
        self.timeframe_choices = ["5min", "1min", "30min", "60min", "1day"]
        self.timeframe_choices_menu = tk.OptionMenu(
            root, self.var_timeframe, *self.timeframe_choices)
        self.label_start_date = tk.Label(root, text="Start Date (YYYY-MM-DD):")
        self.entry_start_date = tk.Entry(root)
        self.entry_start_date.insert(0, self.start_date)
        self.label_end_date = tk.Label(root, text="End Date (YYYY-MM-DD):")
        self.entry_end_date = tk.Entry(root)
        self.entry_end_date.insert(0, self.end_date)
        self.button = tk.Button(root, text="Download and Plot",
                                command=self.download_and_plot)
        self.record_button = tk.Button(
            root, text="Record", command=self.toggle_recording)

        self.test_button = tk.Button(
            root, text="test_button", command=self.download_data)

        # Create a matplotlib Figure and Axis
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Create a FigureCanvasTkAgg widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Pack widgets
        self.label_ticker.pack()
        self.entry_ticker.pack()
        self.label_timeframe.pack()
        self.timeframe_choices_menu.pack()
        self.label_start_date.pack()
        self.entry_start_date.pack()
        self.label_end_date.pack()
        self.entry_end_date.pack()
        self.button.pack()
        self.record_button.pack()
        self.test_button.pack()
        self.canvas_widget.pack()

        # Bind click event to the chart
        self.canvas_widget.bind("<Button-1>", self.on_chart_click)

    def download_and_plot(self):
        self.recording = False  # Stop recording when new data is loaded
        self.entry_ticker.config(state="disabled")
        ticker = self.entry_ticker.get()
        timeframe = self.var_timeframe.get()
        start_date = self.entry_start_date.get()
        end_date = self.entry_end_date.get()

        data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

        self.ax.clear()  # Clear the previous plot
        self.ax.set_title(f"{ticker} Stock Price")
        data["Close"].plot(ax=self.ax)

        self.canvas.draw()  # Redraw the canvas

    def parse_data(self, entry):
        mapped_entry = {
            "Date": pd.to_datetime(entry.timestamp, unit="ms"),
            "Open": entry.open,
            "High": entry.high,
            "Low": entry.low,
            "Close": entry.close,
            "Volume": entry.volume
        }
        return mapped_entry

    def download_data(self):
        ticker = self.entry_ticker.get()
        start_date = self.entry_start_date.get()
        end_date = self.entry_end_date.get()

        # List Aggregates (Bars)
        aggs = []
        for a in self.client.list_aggs(ticker=ticker, multiplier=5, timespan="minute", from_=start_date, to=end_date, limit=50000):
            aggs.append(self.parse_data(a))

        merged_data = pd.DataFrame(aggs)

        csv_filename = os.path.join(
            csv_directory, f"{ticker}_{start_date}_{end_date}.csv")
        merged_data.to_csv(csv_filename)

        self.ax.clear()  # Clear the previous plot
        self.ax.set_title(f"{ticker} Stock Price")
        merged_data["Close"].plot(ax=self.ax)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Close Price")
        self.canvas.draw()  # Redraw the canvas

        return merged_data

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_button.config(text="Stop Recording")
            self.recorded_candles = []
        else:
            self.record_button.config(text="Record")

    def on_chart_click(self, event):
        if self.recording:
            x = event.xdata
            y = event.ydata
            self.recorded_candles.append((x, y))
            print(f"Recorded candle: Time = {x}, Price = {y}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StockDataViewer(root)
    root.mainloop()
