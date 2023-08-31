import pandas as pd
from finta import TA


def load_file(filename):
    from os.path import dirname, join

    # return pd.read_csv(join(dirname(__file__), filename), index_col=0, parse_dates=True, infer_datetime_format=True)
    return pd.read_csv(filename, parse_dates=['Date'])


def load_data(file_name_with_path):
    # Load file
    df = load_file(file_name_with_path)

    # TODO: remove off market history for stock only, pay attentiont to futures
    # Define the time ranges to filter out
    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('21:00:00').time()

    # Filter out off market history
    df = df[(df['Date'].dt.time >= start_time)
            & (df['Date'].dt.time <= end_time)]

    # Rename column Datetime to Date
    """ df = df.rename(
        columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }) """

    # Clear initial NaN values
    df.dropna(inplace=True)

    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create index and sort dataframe
    df.set_index('Date', inplace=True)
    df.sort_values('Date', ascending=True, inplace=True)

    # df["close"] = df["close"]
    df["feature_open"] = df["Open"]
    df["feature_high"] = df["High"]
    df["feature_low"] = df["Low"]
    df["feature_close"] = df["Close"]

    # Create columns for technical indicator features
    df['feature_rsi'] = TA.RSI(df, 12)
    df['feature_vwap'] = TA.VWAP(df)
    df['feature_ema'] = TA.EMA(df, period=50)
    df['feature_atr'] = TA.ATR(df)
    df.dropna(inplace=True)

    return df
