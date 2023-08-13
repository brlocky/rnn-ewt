import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Generate synthetic data (replace with your actual data)
# For demonstration purposes, we're using random data here
data = np.random.rand(500, 5)  # 500 samples, 5 features
labels = np.random.randint(0, 3, size=(500,))  # Three categories

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(data.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Adjust based on the number of wave categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f"Test accuracy: {test_accuracy}")

# Make predictions using the trained model
predictions = model.predict(X_test_scaled)

# Convert predicted probabilities to predicted categories
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
