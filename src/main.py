import numpy as np
from FNN import FeedforwardNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# Define activation functions and their derivatives
def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig

def identity(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x

def mse_loss(y_pred, y_true, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true)  # Gradient of MSE loss
    return np.mean((y_pred - y_true) ** 2)

# Generate training data for sin(x) approximation
X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
Y = np.sin(X)

# Define the FNN model with 1 input neuron, 10 hidden neurons, and 1 output neuron
nn_model = FeedforwardNeuralNetwork(
    layer_sizes=[1, 10, 1],
    activations=[sigmoid, identity],
    learning_rate=0.01,
    num_epochs=500,
    regularization=0
)

# Train the model
nn_model.train(
    X, Y,
    loss_function=mse_loss,
    loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
    batch_size=32
)

# Generate test data for plotting
x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_plot = nn_model.predict(x_plot)

# Plot the results
plt.plot(X, Y, 'b.', label='Training Data (sin(x))')
plt.plot(x_plot, y_plot, 'r-', label='FNN Approximation')
plt.title('FNN Approximation of sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

# Evaluate
test_input = np.array([0.5])
print("Prediction for sin(0.5):", nn_model.predict(test_input))

# Generate training data for Van der Pol system
def van_der_pol_sample(x1, x2):
    dt = 0.5
    x1_next = x1 + x2 * dt
    x2_next = x2 + (-x1 + (1 - x2 ** 2) * x2) * dt
    return np.array([x1_next, x2_next])

X_vdp = np.random.uniform(-3, 3, (1000, 2))
Y_vdp = np.array([van_der_pol_sample(x[0], x[1]) for x in X_vdp])

# Define the FNN model with 2 input neurons, 10 hidden neurons, and 2 output neurons
nn_vdp = FeedforwardNeuralNetwork(
    layer_sizes=[2, 10, 2],
    activations=[sigmoid, identity],
    learning_rate=0.01,
    num_epochs=500,
    regularization=0
)

# Train the model with mini-batch SGD
nn_vdp.train(
    X_vdp, Y_vdp,
    loss_function=mse_loss,
    loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
    batch_size=32
)

# Evaluate
test_input_vdp = np.array([1.0, 1.0])
print("Prediction for Van der Pol step (1.0, 1.0):", nn_vdp.predict(test_input_vdp))
