import pandas as pd


def load_file(filename):
    from os.path import dirname, join

    # return pd.read_csv(join(dirname(__file__), filename), index_col=0, parse_dates=True, infer_datetime_format=True)
    return pd.read_csv(filename, parse_dates=['Date'])


def load_data(file_name_with_path):
    # Load file
    df = load_file(file_name_with_path)

    # Define the time ranges to filter out
    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('21:00:00').time()

    # Filter out off market history
    df = df[(df['Date'].dt.time >= start_time) & (df['Date'].dt.time <= end_time)]

    # Clear initial NaN values
    df.dropna(inplace=True)

    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create index and sort dataframe
    df.set_index('Date', inplace=True)
    df.sort_values('Date', ascending=True, inplace=True)

    return df
