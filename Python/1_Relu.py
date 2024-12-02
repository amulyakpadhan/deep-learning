import numpy as np
def relu(x):
 return np.maximum(0, x)


x = np.array([-1, 0, 1])

# ReLU
print(relu(x)) # [0 0 1]

### Plot ReLU

import numpy as np
import matplotlib.pyplot as plt
def plot_relu():
 # Generate values for x
 x = np.linspace(-10, 10, 100)
 # Compute ReLU values for corresponding x
 relu = np.maximum(0, x)
 # Plot the ReLU function
 plt.plot(x, relu)
 plt.title("ReLU Activation Function")
 plt.xlabel("x")
 plt.ylabel("ReLU(x)")
 plt.grid(True)
 plt.show()

plot_relu()

### Implementation of ReLu in a neural network using the MNIST dataset

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


def relu(x):
 return np.maximum(0, x)


# Build the model
model = keras.Sequential([ layers.Dense(128, input_shape=(28 * 28,), activation='relu'),
 layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)