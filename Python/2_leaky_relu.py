import numpy as np

def leaky_relu(x, alpha=0.01):
 return np.maximum(alpha * x, x)

x = np.array([-1, 0, 1])

# Leaky ReLU
print(leaky_relu(x)) # [-0.01 0. 1. ]

### Plot Leaky ReLU

import numpy as np
import matplotlib.pyplot as plt
def plot_leaky_relu():
 # Generate values for x
 x = np.linspace(-10, 10, 100)
 # Define the leaky ReLU function
 def leaky_relu(x, alpha=0.1):
  return np.where(x >= 0, x, alpha * x)
 # Compute leaky ReLU values for corresponding x
 leaky_relu_values = leaky_relu(x)
 # Plot the leaky ReLU function
 plt.plot(x, leaky_relu_values)
 plt.title("Leaky ReLU Activation Function")
 plt.xlabel("x")
 plt.ylabel("Leaky ReLU(x)")
 plt.grid(True)
 plt.show()

plot_leaky_relu()

### Implementation of leaky relu in a neural network using the MNIST dataset

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

def leaky_relu(x, alpha=0.01):
 return np.maximum(alpha * x, x)

# Build the model with Leaky ReLU
model = keras.Sequential([
    layers.Dense(128, input_shape=(28 * 28,)),
    layers.LeakyReLU(alpha=0.01),  # Apply Leaky ReLU with alpha=0.01
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)