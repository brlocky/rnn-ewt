import numpy as np
import pandas as pd
import taew
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import datetime

# Fetch historical stock data for Apple as an example
stock_symbol = 'AAPL'
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2020, 1, 27)
stock_data = yf.download(stock_symbol, start=start, end=end)

# Calculate the Elliott Wave labels (for example purposes)
# You would need to replace this with your actual wave labels
# Make sure the labels correspond to the data you have
wave_labels = np.random.randint(1, 6, size=len(stock_data))

# Prepare the input features (OHLC) and target labels (waves)
features = stock_data[['Open', 'High', 'Low', 'Close']].values
labels = wave_labels

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(
    X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(5, activation='softmax'))  # 5 classes for wave 1 to 5
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50,
                    batch_size=32, validation_data=(X_test, y_test))

# Save the scaler and model for future use
scaler_filename = 'wave_scaler.pkl'
joblib.dump(scaler, scaler_filename)

model_filename = 'wave_model.h5'
model.save(model_filename)

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot predictions vs. actual labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1) + 1  # Convert back to wave labels
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_labels, alpha=0.5)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Predictions vs. Actual Labels')
plt.tight_layout()

# Save plots to files
plots_filename = 'wave_plots.png'
plt.savefig(plots_filename)

# Load the saved scaler and model
loaded_scaler = joblib.load(scaler_filename)
loaded_model = load_model(model_filename)

# Example prediction using the loaded model and scaler
example_input = X_test[0].reshape(1, X_test.shape[1], X_test.shape[2])
example_input_scaled = loaded_scaler.transform(example_input)
example_prediction = loaded_model.predict(example_input_scaled)
example_wave_label = np.argmax(example_prediction) + 1

print(f'Example Predicted Wave Label: {example_wave_label}')

# Show the plots
plt.show()
