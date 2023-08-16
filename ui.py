import tkinter as tk
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

class StockDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Data Viewer")

        # Set default values
        self.default_symbol = "AAPL"
        self.default_timeframe = "1mo"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.recording = False
        self.recorded_candles = []

        # Create widgets
        self.label_symbol = tk.Label(root, text="Enter Stock Symbol:")
        self.entry_symbol = tk.Entry(root)
        self.entry_symbol.insert(0, self.default_symbol)
        self.label_timeframe = tk.Label(root, text="Select Timeframe:")
        self.var_timeframe = tk.StringVar()
        self.var_timeframe.set(self.default_timeframe)
        self.timeframe_choices = ["1d", "1wk", "1mo"]
        self.timeframe_choices_menu = tk.OptionMenu(root, self.var_timeframe, *self.timeframe_choices)
        self.label_start_date = tk.Label(root, text="Start Date (YYYY-MM-DD):")
        self.entry_start_date = tk.Entry(root)
        self.entry_start_date.insert(0, self.start_date)
        self.label_end_date = tk.Label(root, text="End Date (YYYY-MM-DD):")
        self.entry_end_date = tk.Entry(root)
        self.entry_end_date.insert(0, self.end_date)
        self.button = tk.Button(root, text="Download and Plot", command=self.download_and_plot)
        self.record_button = tk.Button(root, text="Record", command=self.toggle_recording)

        # Create a matplotlib Figure and Axis
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Create a FigureCanvasTkAgg widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Pack widgets
        self.label_symbol.pack()
        self.entry_symbol.pack()
        self.label_timeframe.pack()
        self.timeframe_choices_menu.pack()
        self.label_start_date.pack()
        self.entry_start_date.pack()
        self.label_end_date.pack()
        self.entry_end_date.pack()
        self.button.pack()
        self.record_button.pack()
        self.canvas_widget.pack()

        # Bind click event to the chart
        self.canvas_widget.bind("<Button-1>", self.on_chart_click)

    def download_and_plot(self):
        self.recording = False  # Stop recording when new data is loaded
        self.entry_symbol.config(state="disabled")
        symbol = self.entry_symbol.get()
        timeframe = self.var_timeframe.get()
        start_date = self.entry_start_date.get()
        end_date = self.entry_end_date.get()

        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

        self.ax.clear()  # Clear the previous plot
        self.ax.set_title(f"{symbol} Stock Price")
        data["Close"].plot(ax=self.ax)

        self.canvas.draw()  # Redraw the canvas

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
