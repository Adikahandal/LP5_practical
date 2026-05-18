import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training Images:", X_train.shape)
print("Testing Images:", X_test.shape)

# Normalize Data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Class Labels
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Build CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict Images
predictions = model.predict(X_test[:5])

print("\nPredicted Classes:")

for i in range(5):
    predicted_label = np.argmax(predictions[i])
    print(class_names[predicted_label])

# Display Sample Image
plt.imshow(X_test[0].reshape(28,28), cmap='gray')

plt.title("Predicted: " + class_names[np.argmax(predictions[0])])

plt.show()


"""
What is CNN?

CNN = Convolutional Neural Network

Used mainly for:

image processing
computer vision
What is Convolution?

Feature extraction operation using filters/kernels.

Detects:

edges
shapes
textures
What is Pooling?

Reduces image size and computation.

Used:

MaxPooling2D
Why CNN instead of DNN?

CNN is better for images because:

automatic feature extraction
spatial relationship handling
What is Flatten Layer?

Converts 2D feature maps into 1D vector.

Why Softmax Activation?

Used for multiclass classification.

Outputs probabilities for 10 classes.

Loss Function Used
sparse_categorical_crossentropy

Used because labels are integers.

Model Architecture

Input Image
→ Conv2D
→ MaxPooling
→ Conv2D
→ MaxPooling
→ Flatten
→ Dense
→ Output

Libraries Used
Library	Purpose
TensorFlow/Keras	CNN implementation
NumPy	Numerical operations
Matplotlib	Image display
Important Viva Questions
What is CNN?
Difference between CNN and DNN?
What is convolution?
What is pooling?
Why ReLU is used?
What is flatten layer?
Why CNN is better for images?
What is filter/kernel?
What is softmax activation?
What is feature extraction?
"""