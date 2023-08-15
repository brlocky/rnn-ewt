import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the Excel data
data = pd.read_excel('data.xlsx')

# Define the mapping of wave categories to numerical labels
wave_mapping = {'x': 0, 'wave 1': 1, 'wave 2': 2,
                'wave 3': 3, 'wave 4': 4, 'wave 5': 5}

# Convert 'currentwave' column to numerical labels using the mapping
data['currentwave'] = data['currentwave'].map(wave_mapping)

# Preprocessing the data
features = data[['open', 'high', 'low', 'close', 'volume']].values
labels = data['currentwave'].values

scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

sequence_length = 5

# Specify the number of epochs
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    def create_sequences_and_labels(data, sequence_length):
        sequences = []
        next_wave = []

        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            # Get the last element as label
            label = data[i + sequence_length, -1]

            sequences.append(seq)
            next_wave.append(label)

        return np.array(sequences), np.array(next_wave)

    # Create sequences and labels
    sequences, next_wave = create_sequences_and_labels(
        features_normalized, sequence_length)

    split_ratio = 0.8
    split_idx = int(len(sequences) * split_ratio)

    train_sequences = sequences[:split_idx]
    train_next_wave = next_wave[:split_idx]

    test_sequences = sequences[split_idx:]
    test_next_wave = next_wave[split_idx:]

    num_features = features_normalized.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        train_sequences, train_next_wave, test_size=0.2, random_state=42)

    num_categories = len(np.unique(labels))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(
            sequence_length, num_features), return_sequences=True),
        tf.keras.layers.LSTM(32),
        # Adjust units for the number of unique labels
        tf.keras.layers.Dense(num_categories, activation='softmax')
    ])

    # Load model weights if available
    try:
        model.load_weights('model_weights.h5')
        print("Model weights loaded successfully.")
    except:
        print("No model weights found. Training from scratch.")

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10,
              batch_size=32, validation_split=0.2)

    test_loss, test_acc = model.evaluate(
        X_test, y_test)

    predictions = model.predict(X_test)

    print(predictions)

    predicted_classes = np.argmax(predictions, axis=1)
    print(predicted_classes)

    model.save_weights('model_weights.h5')
