import numpy as np

def softmax(x):
 exp_x = np.exp(x)
 return exp_x / np.sum(exp_x)

x = np.array([-1, 0, 1])

# Softmax
print(softmax(x)) # [0.09003057 0.24472847 0.66524096]


### Plot Softmax

import numpy as np
import matplotlib.pyplot as plt
def plot_softmax(probabilities, class_labels):
 plt.bar(class_labels, probabilities)
 plt.xlabel("Class")
 plt.ylabel("Probability")
 plt.title("Softmax Output")
 plt.show()
# Example usage:
class_labels = ["Class A", "Class B", "Class C"]
probabilities = np.array([0.2, 0.3, 0.5])
plot_softmax(probabilities, class_labels)

### Implementation of softmax in a neural network using the MNIST dataset

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


def softmax(x):
 exp_x = np.exp(x)
 return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Build the model
model = keras.Sequential([ layers.Dense(128, input_shape=(28 * 28,), activation='softmax'),
 layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)