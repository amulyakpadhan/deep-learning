import numpy as np
def sigmoid(x):
 return 1 / (1 + np.exp(-x))

x = np.array([-1, 0, 1])
# Sigmoid
print(sigmoid(x)) # [0.26894142 0.5 0.73105858]


### Plot Sigmoid

import numpy as np
import matplotlib.pyplot as plt
def plot_sigmoid():
 x = np.linspace(-10, 10, 100) # Generate 100 equally spaced values from -10 to 10
 y = 1 / (1 + np.exp(-x)) # Compute the sigmoid function values

 plt.plot(x, y)
 plt.xlabel('Input')
 plt.ylabel('Sigmoid Output')
 plt.title('Sigmoid Activation Function')
 plt.grid(True)
 plt.show()

plot_sigmoid()

### Implementation of sigmoid in a neural network using the MNIST dataset

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

# Define the activation functions
def sigmoid(x):
 return 1 / (1 + np.exp(-x))

# Build the model
model = keras.Sequential([ layers.Dense(128, input_shape=(28 * 28,), activation='sigmoid'),
 layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)