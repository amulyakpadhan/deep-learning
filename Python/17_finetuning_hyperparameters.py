# !pip install keras-tuner

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras_tuner import Hyperband

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to [0, 1] range and reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    
    # Hyperparameter for the number of filters in Conv2D layers
    model.add(Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(
        filters=hp.Int('filters_2', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    # Hyperparameter for the number of units in Dense layers
    model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=10, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Initialize the Keras Tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hyperparameter_tuning',
    project_name='mnist_cnn_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best Hyperparameters:
- Filters in Conv2D layers: {best_hps.get('filters_1')} and {best_hps.get('filters_2')}
- Kernel sizes: {best_hps.get('kernel_size')} and {best_hps.get('kernel_size_2')}
- Dense layer units: {best_hps.get('units')}
- Dropout rate: {best_hps.get('dropout')}
- Optimizer: {best_hps.get('optimizer')}
""")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test data
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")









## 2nd method
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the grid of hyperparameters
filter_options = [32, 64]
kernel_size_options = [3, 5]
units_options = [64, 128]
dropout_options = [0.2, 0.3]
optimizer_options = ['adam', 'sgd']

# Iterate through all combinations of hyperparameters
best_accuracy = 0
best_params = {}

for filters_1 in filter_options:
    for filters_2 in filter_options:
        for kernel_size in kernel_size_options:
            for units in units_options:
                for dropout in dropout_options:
                    for optimizer in optimizer_options:
                        print(f"Training with filters: {filters_1}, {filters_2}, kernel_size: {kernel_size}, units: {units}, dropout: {dropout}, optimizer: {optimizer}")
                        
                        # Define the CNN model
                        model = Sequential([
                            Conv2D(filters=filters_1, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)),
                            MaxPooling2D(pool_size=(2, 2)),
                            Conv2D(filters=filters_2, kernel_size=(kernel_size, kernel_size), activation='relu'),
                            MaxPooling2D(pool_size=(2, 2)),
                            Flatten(),
                            Dense(units=units, activation='relu'),
                            Dropout(dropout),
                            Dense(units=10, activation='softmax')
                        ])
                        
                        # Compile the model
                        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        # Train the model
                        model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=0)
                        
                        # Evaluate the model
                        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                        print(f"Accuracy: {accuracy:.4f}")
                        
                        # Keep track of the best model
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'filters_1': filters_1,
                                'filters_2': filters_2,
                                'kernel_size': kernel_size,
                                'units': units,
                                'dropout': dropout,
                                'optimizer': optimizer
                            }

print(f"Best Hyperparameters: {best_params}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
