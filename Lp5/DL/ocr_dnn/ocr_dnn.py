import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Dataset
data = pd.read_csv("letter-recognition.data", header=None)

# Features and Labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode Labels (A-Z → 0-25)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# One Hot Encoding
y = to_categorical(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(16,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict Sample
predictions = model.predict(X_test[:5])

predicted_classes = np.argmax(predictions, axis=1)

print("\nPredicted Letters:")
print(encoder.inverse_transform(predicted_classes))

# Plot Accuracy Graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.show()


"""
What is Multiclass Classification?

Classification with more than 2 classes.

Example:

A–Z letters
Digits 0–9
Animal categories
Why Softmax Activation?

Used in output layer:

Dense(26, activation='softmax')

Softmax converts outputs into probabilities.

Why 26 Neurons?

Because:

26 English alphabets
What is One-Hot Encoding?

Converts class labels into binary vectors.

Example:

A → [1 0 0 0 ...]
B → [0 1 0 0 ...]
Loss Function Used
categorical_crossentropy

Used for multiclass classification.

Model Architecture

Input(16 Features)
→ Dense(128)
→ Dense(64)
→ Output(26)

Libraries Used
Library	Purpose
TensorFlow/Keras	Deep Learning
Pandas	Dataset handling
NumPy	Numerical operations
Scikit-learn	Preprocessing
Matplotlib	Graph plotting
Important Viva Questions
Difference between binary and multiclass classification?
Why softmax is used?
What is one-hot encoding?
What is categorical crossentropy?
Why feature scaling is needed?
What is activation function?
What is epoch?
What is batch size?
Why ReLU is used?
Difference between sigmoid and softmax?
"""