import numpy as np

def tanh(x):
 return np.tanh(x)

x = np.array([-1, 0, 1])

# Tanh
print(tanh(x)) # [-0.76159416 0. 0.76159416]

### Plot tanh

import numpy as np
import matplotlib.pyplot as plt
def plot_tanh():
 # Generate values for x
 x = np.linspace(-10, 10, 100)
 # Compute tanh values for corresponding x
 tanh = np.tanh(x)
 # Plot the tanh function
 plt.plot(x, tanh)
 plt.title("Hyperbolic Tangent (tanh) Activation Function")
 plt.xlabel("x")
 plt.ylabel("tanh(x)")
 plt.grid(True)
 plt.show()

plot_tanh()

### Implementation of tanh in a neural network using the MNIST dataset

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32') / 255

# Convert labels to one-hot encodings
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def tanh(x):
 return np.tanh(x)

# Build the model
model = keras.Sequential([ layers.Dense(128, input_shape=(28 * 28,), activation='tanh'),
 layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)