import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
import sys


# Calculates classification accuracy (correctly predicted / all values)
def class_acc(pred, gt):
    # How many were correct compared to gt values
    correct = np.sum(pred == gt)
    # Total number of values 
    total = len(gt)
    acc = correct / total *100
    return acc


def class_param(x_train, y_train, num_classes=10):
    means = []
    variances = []
    
    for k in range(num_classes):
        # Get all samples of class k
        x_class_k = x_train[y_train == k]
        
        # Compute mean and variance for each dimension i.e. pixel
        mean_k = np.mean(x_class_k, axis=0)
        var_k = np.var(x_class_k, axis=0) + 0.001
        
        means.append(mean_k)
        variances.append(var_k)
    
    return np.array(means), np.array(variances)


def log_probab(x_test, means, variances, num_classes=10):
    log_probabs = np.zeros((x_test.shape[0], num_classes))

    for k in range(num_classes):
        
        # Take the mean and variance of class k
        mean_k = means[k]
        var_k = variances[k]
        # Use log[p(class|input vector)] 
        term1 = -0.5 * np.sum(np.log(2 * np.pi * var_k))
        term2 = -0.5 * np.sum(((x_test - mean_k) ** 2) / var_k, axis=1)

        log_probabs[:, k] = term1 + term2
    
    return log_probabs


def main():

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

    # Calculate mean and variance for each pixel in each class
    means, variances = class_param(x_train_reshaped, y_train)

    # Calculate logarithmic probabilities for each class given an input vector
    probabilities = log_probab(x_test_reshaped, means, variances)
    # Find highest probabilities (out of 10) for each picture 
    pred = np.argmax(probabilities, axis=1)
    
    acc = class_acc(pred, y_test)
    print(f"Classification accuracy is {acc} %")


if __name__ == "__main__":
    main()
    