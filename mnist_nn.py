import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys

# Calculates classification accuracy (correctly predicted / all values)
def class_acc(pred, gt):
    # How many were correct compared to gt values
    correct = np.sum(pred == gt)
    # Total number of values 
    total = len(gt)
    acc = correct / total *100
    return acc

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

# Use 1-NN classifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train_reshaped, y_train)
pred = neigh.predict(x_test_reshaped)
# Calculate prediction accuracy
acc = class_acc(pred, y_test)
print(f"Classification accuracy is {acc} %")
