import numpy as np

from FNN import FeedforwardNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#Store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def mse_loss(y_pred, y_true, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / len(y_true)
    return np.mean((y_pred - y_true) ** 2)


def logsoftmax(x, derivative=False):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    log_softmax = np.log(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

    if derivative:
        softmax_x = np.exp(log_softmax)
        return softmax_x * (1 - softmax_x)
    return log_softmax


def NNN_loss(y_pred, y_true, derivative=False):

    if derivative:
        grad = np.copy(y_pred)
        grad[range(len(y_pred)), y_true] -= 1
        return grad / len(y_pred)  # Gradient with respect to the input

    return np.mean(-y_pred[range(len(y_pred)), y_true])

def identity(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x

# Hand written characters
# https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0 # normalize to 0 to 1

# Split data into train partition and test partition

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

# sample down from 21000 to x data points
num_samples = 1000
X_train_sampled = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
y_train_sampled = y_train[np.random.choice(y_train.shape[0], 1000, replace=False)]
X_test_sampled = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
Y_test_sampled = y_test[np.random.choice(y_test.shape[0], 1000, replace=False)]


# Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron

nn = FeedforwardNeuralNetwork(layer_sizes=[784, 64, 10], activations=[sigmoid, identity])

print("traing")
nn.train(X_train, y_train, loss_function=NNN_loss, loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
         epochs=10, learning_rate=0.01)
print("test")

def test_handwritten(self, X_test, y_test):
    y_pred = nn.forward(X_test)

    correct = 0
    incorrect = 0
    for i in range(len(y_test)):
        if(y_test[i] == y_pred[i]):
            correct += 1
        else:
            incorrect += 1
    print("correct:", correct, " incorrect:", incorrect)