
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import pandas as pd

# Define the column names
feature_columns = ['Open', 'High', 'Low',
                   'Close', 'HighLow', 'CloseOpen', 'TrendLabel']


def enrich_data(file_path):
    # Read the csv file
    df = pd.read_csv(file_path)

    labeled_data = []

    for i in range(2, len(df)):
        high = df['High'][i]
        low = df['Low'][i]
        open_price = df['Open'][i]
        close = df['Close'][i]

        if close > open_price:
            trend = 1
        elif open_price > close:
            trend = -1
        else:
            trend = 0

        labeled_row = {
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'HighLow': high - low,
            'CloseOpen': close - open_price,
            'TrendLabel': trend
        }
        labeled_data.append(labeled_row)

    # Create a new DataFrame with the enriched data
    df_for_training = pd.DataFrame(labeled_data)

    # Clear na
    df_for_training.fillna(0, inplace=True)

    return df, labeled_data, df_for_training, feature_columns


def prepare_training_data(df_for_training_scaled):
    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 9. We will make timesteps = 30 (past days data used for training).

    # Empty lists to be populated using formatted training data
    trainX = []
    trainY = []

    # Number of days we want to look into the future based on the past days.
    n_future = 16
    n_past = 15  # Number of past days we want to use to predict the future.

    # Reformat input data into a shape: (n_samples x timesteps x n_features)
    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(
            df_for_training_scaled[i - n_past:i, :])  # Include all columns in trainX

        close_index = feature_columns.index('Close')
        close_open_index = feature_columns.index('CloseOpen')

        trainY.append([
            # 'Close' column index
            df_for_training_scaled[i + n_future - 1:i + \
                                   n_future, close_index],
            # 'CloseOpen' column index
            df_for_training_scaled[i + n_future - \
                                   1:i + n_future, close_open_index]
        ])

    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY


def predict(model, df, trainX, trainY, df_for_training_scaled, scaler):
    print("Predicting...")
    # Predicting...
    # Libraries that will help us extract only business days in the US.
    # Otherwise our dates would be wrong when we look back (or forward).
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # Remember that we can only predict one day in the future as our model needs 5 variables
    # as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    n_past = 16
    n_days_for_prediction = 15  # number of days to feed the network

    # Separate dates for future plotting
    train_dates = pd.to_datetime(df['Date'])

    # Generate future prediction dates
    predict_period_dates = pd.date_range(
        list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()

    print('predict_period_dates', len(
        predict_period_dates), predict_period_dates)
    # normalize the prediction
    # scaler = StandardScaler()
    # Reshape trainY to 2D array
    # trainY_reshaped = trainY.reshape(-1, 2)
    # scaler = scaler.fit(trainY_reshaped)

    # Prepare training data
    trainX, trainY = prepare_training_data(df_for_training_scaled)

    # Make prediction using the model
    prediction = model.predict(trainX[-n_days_for_prediction:])

    # Create dummy columns for missing features
    num_missing_features = 7 - prediction.shape[1]
    dummy_columns = np.zeros((prediction.shape[0], num_missing_features))

    # Concatenate the prediction and dummy columns
    prediction_with_dummy = np.concatenate((prediction, dummy_columns), axis=1)

    # Inverse transform using the original scaler
    predicted_features = scaler.inverse_transform(prediction_with_dummy)

    # Extract the predicted 'CloseOpen' and 'Close' features
    y_pred_future_closeopen = predicted_features[:, 1]
    y_pred_future_close = predicted_features[:, 0]

    print('y_pred_future_close', y_pred_future_close)
    print('y_pred_future_closeopen', y_pred_future_closeopen)

    # Convert timestamp to date
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())

    # Create a DataFrame for the forecasted values
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Close': y_pred_future_close,
        'CloseOpen': y_pred_future_closeopen
    })

    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    return df_forecast
