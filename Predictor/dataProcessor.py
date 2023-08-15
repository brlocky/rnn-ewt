import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Define the column names
feature_columns = ['Open', 'High', 'Low',
                   'Close', 'Volume', 'HighLow', 'CloseOpen', 'TrendLabel']


class MarketDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler()
        self.original_data = None
        self.training_data = None
        self.trainX = None
        self.trainY = None
        self.prepare_data()

    def prepare_data(self):
        # Load your data from the file

        # Prepare Data
        self.enrich_data()

        # Prepare training data
        self.prepare_training_data()

    def enrich_data(self):
        # Read the csv file
        df = pd.read_csv(self.file_path)

        labeled_data = []

        for i in range(2, len(df)):
            high = df['High'][i]
            low = df['Low'][i]
            open = df['Open'][i]
            close = df['Close'][i]

            if close > open:
                trend = 1
            elif open > close:
                trend = -1
            else:
                trend = 0

            labeled_row = {
                'Date': df['Date'][i],  # Include the 'Date' column
                'Open': open,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': df['Volume'][i],
                'HighLow': high - low,
                'CloseOpen': close - open,
                'TrendLabel': trend
            }

            labeled_data.append(labeled_row)

        # Create a new DataFrame with the enriched data
        df_original = pd.DataFrame(labeled_data)

        # Clear na
        df_original.fillna(0, inplace=True)

        df_original['Date'] = pd.to_datetime(df_original['Date'])
        df_training = df_original.drop(columns=['Date'])

        # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        self.scaler = self.scaler.fit(df_training)
        self.original_data = df_original
        self.training_data = self.scaler.transform(df_training)

    def prepare_training_data(self):
        # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
        # In this example, the n_features is 9. We will make timesteps = 30 (past days data used for training).

        # Empty lists to be populated using formatted training data
        trainX = []
        trainY = []

        # Number of days we want to look into the future based on the past days.
        n_future = 16
        # Number of past days we want to use to predict the future.
        n_past = 15

        # Reformat input data into a shape: (n_samples x timesteps x n_features)
        for i in range(n_past, len(self.training_data) - n_future + 1):
            trainX.append(
                self.training_data[i - n_past:i, :])  # Include all columns in trainX

            close_index = feature_columns.index('Close')
            # close_open_index = feature_columns.index('CloseOpen')

            trainY.append([
                # 'Close' column index
                self.training_data[i + n_future - 1:i + \
                                   n_future, close_index],
                # 'CloseOpen' column index
                # df_for_training_scaled[i + n_future - \
                #                      1:i + n_future, close_open_index]
            ])

        trainX, trainY = np.array(trainX), np.array(trainY)

        # print('trainx', trainX)
        # print('trainY', trainY)
        self.trainX = trainX
        self.trainY = trainY
