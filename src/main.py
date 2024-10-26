import time

from FNN import FeedforwardNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
from scipy.integrate import odeint
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#Store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    x = np.clip(x, -100, 100) #prevent weights from overflowing
    x += + 1e-10 #prevent divide by 0
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def mse_loss(y_pred, y_true, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / len(y_true)
    return np.mean((y_pred - y_true) ** 2)


def logsoftmax(x, derivative=False):
    x = np.clip(x, -100, 100) #prevent weights from overflowing
    x += + 1e-10 #prevent divide by 0
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

run_sin = False
run_vander = False
run_digits = True


if(run_sin):
    X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
    Y = np.sin(X)

    # Define the FNN model with 1 input neuron, 10 hidden neurons, and 1 output neuron
    nn = FeedforwardNeuralNetwork(layer_sizes=[1, 30, 1], activations=[sigmoid, identity])

    # Train the model
    nn.train(X, Y, loss_function=mse_loss, loss_derivative=lambda y_pred, y_true: mse_loss(y_pred, y_true, derivative=True),
             epochs=100, learning_rate=0.05)

    # Generate test data for plotting
    x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = nn.forward(x_plot)

    # Plot the results
    plt.plot(X, Y, 'b.', label='Training Data (sin(x))')
    plt.plot(x_plot, y_plot, 'r-', label='FNN Approximation')
    plt.title('FNN Approximation of sin(x)')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()


def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0]**2) * x[1]]

def Phi(x):
    t = np.linspace(0, 0.05, 101)
    sol = odeint(ode_model, x, t)
    return sol[-1]

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hd1 = torch.nn.Linear(2, 64)
        self.hd2 = torch.nn.Linear(64, 64)
        self.hd3 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 2)

    def forward(self, x):
        a1 = self.hd1(x)
        h1 = torch.nn.functional.relu(a1)
        a2 = self.hd2(h1)
        h2 = torch.nn.functional.relu(a2)
        a3 = self.hd3(h2)  # Corrected from self.hd2 to self.hd3
        h3 = torch.nn.functional.relu(a3)
        y = self.output(h3)
        return y

# VANDER_POL
if(run_vander):
    net = Net()

    # Compute the samples
    # X is a set of samples in a 2D plane
    # Y consists of the corresponding outputs of the samples in X
    N = 101  # number of samples in each dimension
    samples_x1 = torch.linspace(-3, 3, N)
    samples_x2 = torch.linspace(-3, 3, N)
    X = torch.empty((0, 2))

    for x1 in samples_x1:
        for x2 in samples_x2:
            sample_x = torch.Tensor([[x1, x2]])
            X = torch.cat((X, sample_x))

    Y = torch.empty((0, 2))
    for x in X:
        y = Phi(x)
        sample_y = torch.Tensor([[y[0], y[1]]])
        Y = torch.cat((Y, sample_y))

    hat_Y = net(X)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(600):
        hat_Y = net(X)
        loss = loss_fn(hat_Y, Y)
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Test 1
    x0 = [1.25, 2.35]
    for i in range(150):
        y = Phi(x0)
        plt.plot(y[0], y[1], 'b.')
        x0 = y
        x0 = torch.Tensor(x0)

    for i in range(150):
        y = net(x0)
        np_y = y.data.numpy()
        plt.plot(np_y[0], np_y[1], 'r.')
        x0 = y

    plt.xlabel('x_1')
    plt.ylabel('x_2')

    # Test 2
    plt.show()

if(run_digits):
    LEARNING_RATE = 1
    EPOCHS = 30
    TEST_SIZE = 10000
    layer_sizes = [784, 128, 128, 10]

    # Hand written characters
    # https://www.openml.org/d/554
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,parser="liac-arff")
    X = X / 255.0 # normalize to 0 to 1

    # Split data into train partition and test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    #convert strings to ints
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # random sample of test data
    sample_indices_test = np.random.choice(X_test.shape[0], TEST_SIZE, replace=False)
    X_test = X_test[sample_indices_test]
    y_test = y_test[sample_indices_test]

    # Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron

    nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid,sigmoid,sigmoid])

    print("training")
    print(f"Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}")
    start_time = time.time()  # Record start time

    nn.train(X_train, y_train, loss_function=NNN_loss,
             loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
             epochs=EPOCHS, learning_rate=LEARNING_RATE)

    end_time = time.time()  # Record end time
    training_duration = np.round((end_time - start_time) / 60)  # Calculate duration

    print("test")

    correct_counts = np.zeros(10)
    incorrect_counts = np.zeros(10)

    for i in range(len(y_test)):
        y_pred = nn.forward(X_test[i])
        if np.random.randint(1, 1000) == 1:
                print(y_pred)
        if np.argmax(y_pred)  ==  y_test[i]:
            correct_counts[y_test[i]] += 1
        else:
            incorrect_counts[y_test[i]] += 1

    print("correct: ", np.sum(correct_counts), "incorrect: ", np.sum(incorrect_counts))
    digits = np.arange(10)

    # Plotting correct vs incorrect classifications for each digit
    plt.figure(figsize=(10, 6))
    plt.bar(digits, correct_counts, label="Correct", color="blue")
    plt.bar(digits, incorrect_counts, bottom=correct_counts, label="Incorrect", color="red")

    # Labeling the plot
    plt.xlabel("Digit")
    plt.ylabel("Number of Classifications")
    plt.title(f"Correct vs Incorrect | Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}, T:{training_duration} min,Correct: {((np.sum(correct_counts)/len(y_test)) * 100):.2f}%")
    plt.xticks(digits)
    plt.legend()

    # Show plot
    plt.show()

