#  Boston housing dataset code

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load Boston Housing Dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network Model
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile Model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2
)

# Evaluate Model
loss, mae = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Mean Absolute Error:", mae)

# Predict House Prices
predictions = model.predict(X_test[:5])

print("\nPredicted Prices:")
print(predictions.flatten())

print("\nActual Prices:")
print(y_test[:5])

# Plot Graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()






'''

# California Housing dataset, it is used because the newer versions don't support the boston housing dataset due to ethical issues
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=32
)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Mean Absolute Error:", mae)

# Predict sample values
predictions = model.predict(X_test[:5])

print("\nPredicted Prices:")
print(predictions.flatten())

print("\nActual Prices:")
print(y_test[:5])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()




'''




"""
What is Regression?

Regression predicts continuous numerical values.

Example:

House price prediction
Temperature prediction
Why DNN?

Deep Neural Network learns complex relationships between:

house features
price
What is Feature Scaling?

Scaling converts data into similar range for better training.

Used:

StandardScaler()
What is Loss Function?

Used:

mse

MSE = Mean Squared Error

Measures prediction error.

What is MAE?

MAE = Mean Absolute Error

Average absolute difference between:

predicted values
actual values
Layers Used
Dense Layer

Fully connected neural network layer.

ReLU Activation

Removes negative values and improves learning.

Model Architecture

Input → Dense(64) → Dense(32) → Output(1)

Libraries Used
Library	Purpose
TensorFlow/Keras	Deep Learning
NumPy	Numerical operations
Pandas	Data handling
Matplotlib	Graph plotting
Scikit-learn	Dataset & preprocessing
Possible Viva Questions
Difference between classification and regression?
Why feature scaling is needed?
What is optimizer?
What is Adam optimizer?
What is epoch?
What is batch size?
Why ReLU is used?
What is overfitting?
What is validation data?
Why output layer has 1 neuron?

"""

