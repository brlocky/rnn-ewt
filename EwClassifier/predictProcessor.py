import os
import pandas as pd
from utils_plot import print_plot_prediction_waves, plot_candlestick_chart


class PredictProcessor:
    def __init__(self, model, data, name, output):
        self.model = model
        self.data = data
        self.name = name
        self.output = output
        self.predict()

    def predict(self):
        n_days_for_prediction = len(self.data.original_data) - 1

        # Make prediction using the model
        prediction = self.model.predict(
            self.data.trainX[-n_days_for_prediction:])

        # Generate future prediction dates
        forecast_dates = self.data.original_data['Date'].iloc[-n_days_for_prediction:]

        # Create a DataFrame for the forecasted values
        df_forecast = pd.DataFrame({
            'Date': forecast_dates[-len(prediction):],
            # Assuming wave 1 prediction is in the first column
            'Wave1': prediction[:, 0],
            # Assuming wave 2 prediction is in the second column
            'Wave2': prediction[:, 1],
            # Assuming wave 3 prediction is in the third column
            'Wave3': prediction[:, 2],
            # Assuming wave 4 prediction is in the fourth column
            'Wave4': prediction[:, 3],
            # Assuming wave 5 prediction is in the fifth column
            'Wave5': prediction[:, 4],
        })

        # Filter original data to keep only dates after '2023-01-01'
        df_original = self.data.original_data[-n_days_for_prediction:].copy()

        # Convert the 'Date' column to datetime
        df_original['Date'] = pd.to_datetime(df_original['Date'])
        df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

        # Save the plot as an image file in the output folder
        output_filename = os.path.join(
            self.output, self.name + '_ew_plot.png')
        print_plot_prediction_waves(
            output_filename, df_original, df_forecast)

        output_filename = os.path.join(
            self.output, self.name + '_chandle_plot.png')
        plot_candlestick_chart(self.data.original_data, output_filename)
