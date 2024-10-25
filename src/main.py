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

#
# X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
# Y = np.sin(X)
#
# # Define the FNN model with 1 input neuron, 10 hidden neurons, and 1 output neuron
# nn = FeedforwardNeuralNetwork(layer_sizes=[1, 30, 1], activations=[sigmoid, identity])
#
# # Train the model
# nn.train(X, Y, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
#          epochs=1000, learning_rate=0.05)
#
# # Generate test data for plotting
# x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
# y_plot = nn.forward(x_plot)
#
# # Plot the results
# plt.plot(X, Y, 'b.', label='Training Data (sin(x))')
# plt.plot(x_plot, y_plot, 'r-', label='FNN Approximation')
# plt.title('FNN Approximation of sin(x)')
# plt.xlabel('x')
# plt.ylabel('sin(x)')
# plt.legend()
# plt.show()
#
#
# # Generate training data
# X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
# Y = np.sin(X)
#
# # Define the FNN model with 1 input neuron, 10 hidden neurons, and 1 output neuron
#
# nn = FeedforwardNeuralNetwork(layer_sizes=[1, 10, 1], activations=[sigmoid, identity])
#
#
# # Train the model
# nn.train(X, Y, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
#          epochs=10, learning_rate=0.01)
#
# # Evaluate
# test_input = np.array([[0.5]])
# print("Prediction for sin(0.5):", nn.forward(test_input))
#
# # Generate training data for Van der Pol system
# def van_der_pol_sample(x1, x2):
#     dt = 0.5
#     x1_next = x1 + x2 * dt
#     x2_next = x2 + (-x1 + (1 - x2 ** 2) * x2) * dt
#     return np.array([x1_next, x2_next])
#
# X_vdp = np.random.uniform(-3, 3, (1000, 2))
# Y_vdp = np.array([van_der_pol_sample(x[0], x[1]) for x in X_vdp])
#
# # Define the FNN model with 2 input neurons, 10 hidden neurons, and 2 output neurons
# nn_vdp = FeedforwardNeuralNetwork(layer_sizes=[2, 10, 2], activations=[sigmoid, identity])
#
#
# # Train the model with mini-batch SGD
# nn_vdp.train(X_vdp, Y_vdp, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
#              epochs=10, learning_rate=0.01, batch_size=32)
#
# # Evaluate
# test_input_vdp = np.array([[1.0, 1.0]])
# print("Prediction for Van der Pol step (1.0, 1.0):", nn_vdp.forward(test_input_vdp))
#

# Hand written characters
# https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,parser="liac-arff")
X = X / 255.0 # normalize to 0 to 1

# Split data into train partition and test partition

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

#convert strings to ints
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# sample down from 21000 to x data points
num_samples = 100
sample_indices_train = np.random.choice(X_train.shape[0], num_samples, replace=False)
X_train_sampled = X_train[sample_indices_train]
y_train_sampled = y_train[sample_indices_train]

sample_indices_test = np.random.choice(X_test.shape[0], num_samples, replace=False)
X_test_sampled = X_test[sample_indices_test]
y_test_sampled = y_test[sample_indices_test]

# Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron

nn = FeedforwardNeuralNetwork(layer_sizes=[784, 64, 64, 10], activations=[logsoftmax,logsoftmax, identity])

print("training")
nn.train(X_train_sampled, y_train_sampled, loss_function=NNN_loss,
         loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
         epochs=1, learning_rate=0.1)

print("test")

# TODO: get the nn to print real values

def test_handwritten(X_test, y_test):
    y_pred = nn.forward(X_test)

    print(y_pred)

    correct = 0
    incorrect = 0
    for i in range(len(y_test)):
        if(y_test[i] == np.argmax(y_pred[i])):
            correct += 1
        else:
            incorrect += 1
    print("correct:", correct, " incorrect:", incorrect)

test_handwritten(X_test_sampled, y_test_sampled)
