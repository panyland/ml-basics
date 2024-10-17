import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, optimizers
import keras
from random import random
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
sys.stdout.reconfigure(encoding='utf-8')


# Select dataset depending on provided arguments 
mnist = tf.keras.datasets.mnist
if len(sys.argv) > 1:
    if sys.argv[1] == "fashion":
        mnist = tf.keras.datasets.fashion_mnist

# Load training and testing data from database
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape data to matrices of 60000x784
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

# Scale values to [0,1]
x_train_reshaped = x_train_reshaped / 255
x_test_reshaped = x_test_reshaped / 255

# One-hot encoding for training and testing labels
encoded_y_train = np.zeros((y_train.size, y_train.max()+1), dtype=int)
encoded_y_train[np.arange(y_train.size),y_train] = 1
encoded_y_test = np.zeros((y_test.size, y_test.max()+1), dtype=int)
encoded_y_test[np.arange(y_test.size),y_test] = 1

# Build model
model = models.Sequential() 
model.add(layers.Dense(128, activation='relu', input_shape=(784,))) 
model.add(layers.Dense(64, activation='relu'))  
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
tr_hist = model.fit(x_train_reshaped, encoded_y_train, epochs=10)

# Test model
test_loss, test_accuracy = model.evaluate(x_test_reshaped, encoded_y_test)
print('Test accuracy:', test_accuracy)

# Plot loss
plt.plot(tr_hist.history['loss'])
plt.title('Loss as a function of epochs')
plt.show()
