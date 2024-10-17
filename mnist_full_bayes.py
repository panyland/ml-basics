from scipy.stats import multivariate_normal
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


# Add gaussian noise to input data
def add_noise(x_train, scale=0.1):
    noise = np.random.normal(loc=0.0, scale=scale, size=x_train.shape)
    return x_train + noise


def class_param(x_train, y_train, num_classes=10):
    means = []
    covariances = []
    
    for k in range(num_classes):
        # Get all samples of class k
        x_class_k = x_train[y_train == k]
        
        # Compute mean and covariance matrix for class k
        mean_k = np.mean(x_class_k, axis=0)
        var_k = np.cov(x_class_k, rowvar=False)
        
        means.append(mean_k)
        covariances.append(var_k)
    
    return np.array(means), np.array(covariances)


def log_multidim_probab(x_test, means, covariances, num_classes=10):
    multidim_probabs = np.zeros((x_test.shape[0], num_classes))

    for k in range(num_classes):
        
        # Take the mean and variance of class k
        mean_k = means[k]
        covar_k = covariances[k]
        # Find distribution and probabilities for test data
        dist = multivariate_normal.pdf(mean_k, covar_k)
        multidim_probabs[:, k] = dist.pdf(x_test)
    
    return multidim_probabs


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
    # Add noise for non-zero variances
    x_train_noisy = add_noise(x_train_reshaped)
    print(x_train_noisy.shape)
    # Calculate mean and variance for each pixel in each class
    means, covariances = class_param(x_train_noisy, y_train)

    # Calculate probabilities for each class given an input vector
    probabilities = log_multidim_probab(x_test_reshaped, means, covariances)
    # Find highest probabilities (out of 10) for each picture 
    pred = np.argmax(probabilities, axis=1)
    
    acc = class_acc(pred, y_test)
    print(f"Classification accuracy is {acc} %")


if __name__ == "__main__":
    main()