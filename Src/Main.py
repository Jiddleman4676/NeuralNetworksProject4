import numpy as np
from FNN import FeedforwardNeuralNetwork

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def mse_loss(y_pred, y_true, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / len(y_true)
    return np.mean((y_pred - y_true) ** 2)
def identity(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x


# Generate training data
X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
Y = np.sin(X)

# Define the FNN model with 1 input neuron, 10 hidden neurons, and 1 output neuron

nn = FeedforwardNeuralNetwork(layer_sizes=[1, 10, 1], activations=[sigmoid, identity])


# Train the model
nn.train(X, Y, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
         epochs=10, learning_rate=0.01)

# Evaluate
test_input = np.array([[0.5]])
print("Prediction for sin(0.5):", nn.forward(test_input))

# Generate training data for Van der Pol system
def van_der_pol_sample(x1, x2):
    dt = 0.5
    x1_next = x1 + x2 * dt
    x2_next = x2 + (-x1 + (1 - x2 ** 2) * x2) * dt
    return np.array([x1_next, x2_next])

X_vdp = np.random.uniform(-3, 3, (1000, 2))
Y_vdp = np.array([van_der_pol_sample(x[0], x[1]) for x in X_vdp])

# Define the FNN model with 2 input neurons, 10 hidden neurons, and 2 output neurons
nn_vdp = FeedforwardNeuralNetwork(layer_sizes=[2, 10, 2], activations=[sigmoid, identity])


# Train the model with mini-batch SGD
nn_vdp.train(X_vdp, Y_vdp, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
             epochs=10, learning_rate=0.01, batch_size=32)

# Evaluate
test_input_vdp = np.array([[1.0, 1.0]])
print("Prediction for Van der Pol step (1.0, 1.0):", nn_vdp.forward(test_input_vdp))
