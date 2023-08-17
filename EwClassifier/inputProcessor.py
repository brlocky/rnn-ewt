import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Define the column names
original_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
label_columns = ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5']


class InputProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.original_data = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.n_past = 5
        self.prepare_data()

    def prepare_data(self):
        # Load the data
        df = pd.read_csv(self.file_path)

        # Extract Original Columns
        df_original = df[original_columns].copy()
        df_original['Date'] = pd.to_datetime(df_original['Date'])
        self.original_data = df_original

        # Separate features (Open, High, Low, Close) and labels (Wave1 to Wave5)
        features = df[feature_columns].values

        try:
            # Check if all label_columns exist in the DataFrame
            labels = df[label_columns].values
        except KeyError as e:
            labels = None

        # Split the data into training and testing sets
        if labels is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2)
        else:
            X_train, X_test = train_test_split(
                features, test_size=0.2, shuffle=False)
            y_train, y_test = None, None

        # Normalize the data using a StandardScaler
        # self.scaler = StandardScaler()
        # X_train_scaled = self.scaler.fit_transform(X_train)
        # X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = X_train
        X_test_scaled = X_test

        # Create sequences for training data
        X_train_sequences, y_train_sequences = self.create_sequences(
            X_train_scaled, y_train, self.n_past)

        # Create sequences for testing data
        X_test_sequences, y_test_sequences = self.create_sequences(
            X_test_scaled, y_test, self.n_past)

        self.trainX = X_train_sequences
        self.trainY = y_train_sequences

        self.testX = X_test_sequences
        self.testY = y_test_sequences

    # Define the create_sequences function
    def create_sequences(self, data, target, length):
        sequences = []
        targets = []

        for i in range(length, len(data)):
            sequence = data[i - length:i]
            if target is not None:
                label = target[i - 1]
                targets.append(label)

            sequences.append(sequence)

        return np.array(sequences), np.array(targets)
