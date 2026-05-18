import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

# Load IMDB Dataset
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Pad sequences
max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Build Deep Neural Network
model = Sequential()

model.add(Embedding(vocab_size, 32, input_length=max_length))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict Reviews
predictions = model.predict(X_test[:5])

print("\nPredictions:")
for i in predictions:
    if i > 0.5:
        print("Positive Review")
    else:
        print("Negative Review")

# Plot Accuracy Graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.show()




"""
What is Binary Classification?

Classification with only 2 classes.

Example:

Positive / Negative
Spam / Not Spam
Why Sigmoid Activation?

Used in output layer:

Dense(1, activation='sigmoid')

Produces output between:

0 and 1
What is Embedding Layer?

Converts words into dense numerical vectors for neural network processing.

Why Padding is Used?

Reviews have different lengths.

Padding makes all reviews same size.

Loss Function Used
binary_crossentropy

Used for binary classification problems.

Model Architecture

Input Reviews
→ Embedding Layer
→ Flatten
→ Dense(64)
→ Output(1)

Libraries Used
Library	Purpose
TensorFlow/Keras	Deep Learning
NumPy	Numerical operations
Matplotlib	Graph plotting
Important Viva Questions
What is sentiment analysis?
Difference between binary and multiclass classification?
Why sigmoid activation is used?
What is embedding layer?
What is padding?
What is binary crossentropy?
Why text preprocessing is needed?
What is vocabulary size?
What is overfitting?
What is accuracy?
"""