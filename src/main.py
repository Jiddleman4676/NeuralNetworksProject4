import ssl
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from FNN import FeedforwardNeuralNetwork
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
import random

# Test (Andrei)
ssl._create_default_https_context = ssl._create_unverified_context

# Store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    x = np.clip(x, -100, 100) # Prevent weights from overflowing
    x += + 1e-10 # Prevent divide by 0
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

# Mean squared error (MSE) loss function with optional derivative
def mse_loss(y_pred, y_true, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true) / len(y_true)
    return np.mean((y_pred - y_true) ** 2)

# Log softmax activation with optional derivative
def logsoftmax(x, derivative=False):
    x = np.clip(x, -100, 100) # Prevent weights from overflowing
    x += + 1e-10 # Prevent divide by 0
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    log_softmax = np.log(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

    if derivative:
        softmax_x = np.exp(log_softmax)
        return softmax_x * (1 - softmax_x)
    return log_softmax

# Custom loss function with optional derivative
def NNN_loss(y_pred, y_true, derivative=False):
    if derivative:
        grad = np.copy(y_pred)
        grad[range(len(y_pred)), y_true] -= 1
        return grad / len(y_pred)  # Gradient with respect to the input

    return np.mean(-y_pred[range(len(y_pred)), y_true])

# Identity activation function with optional derivative
def identity(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x
def mse_loss_derivative(inputV, val):
    return mse_loss(inputV, val, derivative=True)

# Flags for different experiments
run_sin = True
run_vander = True
run_digits = True

# Sinusoidal regression experiment
if(run_sin):
    min = -3
    max = 3
    training_samples = 1000
    test_samples = 200
    num_epochs = 500
    the_learning_rate = 0.01
    hiddenLayers = 10
    #generates the random samples for x values
    X = np.random.uniform(min, max, training_samples).reshape(-1, 1)
    #make our Y values for training
    Y = np.sin(X)



    # Make our ffn, layer sizes are input, hidden layers, and then output
    nn = FeedforwardNeuralNetwork(layer_sizes=[1, hiddenLayers, 1], activations=[sigmoid, identity])


    cur_loss = mse_loss
    cur_loss_deriv = mse_loss_derivative

    #Training using the FNN class
    nn.train(
        X,
        Y,
        loss_function=cur_loss,
        loss_derivative=cur_loss_deriv,

        epochs=num_epochs,
        learning_rate=the_learning_rate
    )



    #Ploting data from test
    test_min = -3
    test_max = 3
    x_plot = np.linspace(test_min, test_max, test_samples).reshape(-1, 1)
    y_plot = nn.forward(x_plot)

    # Plot the results
    plt.plot(X, Y, 'g.', label='Real exact sin(x) values used for training')
    plt.plot(x_plot, y_plot, 'b-', label='Our approximation from our FNN')
    plt.title('Our Approximation of sin(x) using the FFN')
    plt.xlabel('x input')
    plt.ylabel('sin(x) function value')
    plt.legend()
    plt.show()

# ODE model for generating data
def ode_model(x, t):
    # This is what it says on the document, graph is a little weird. Want to check if it is intended
    return [x[1], -x[0] + (1 - x[1]**2) * x[1]]

# Solves ODE model for a given input
def Phi(x):
    t = np.linspace(0, 0.5, 2)
    sol = odeint(ode_model, x, t)
    return sol[-1]

# Torch neural network model for VanderPol
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

# Van der Pol oscillator experiment
if(run_vander):
    # Generate samples in the state space
    N = 101  # Number of samples in each dimension
    samples_x1 = torch.linspace(-3, 3, N)
    samples_x2 = torch.linspace(-3, 3, N)
    X = torch.empty((0, 2))

    for x1 in samples_x1:
        for x2 in samples_x2:
            sample_x = torch.Tensor([[x1, x2]])
            X = torch.cat((X, sample_x))

    # Generate target values using Phi function
    Y = torch.empty((0, 2))
    for x in X:
        y = Phi(x)
        sample_y = torch.Tensor([[y[0], y[1]]])
        Y = torch.cat((Y, sample_y))

    # Initialize the neural network
    net = Net()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    #Part for batches
    batch_num = 32
    num_epochs = 200

    # Training loop with mini-batch SGD
    for epoch in range(num_epochs):

        size_of_dim = 0
        indices = list(range(X.size(size_of_dim)))
        random.shuffle(indices)


        the_X_vals = X[indices]
        the_Y_vals = Y[indices]

        #finding where to end the batch here
        dim_size = X.size(0)
        total_batches = batch_num
        batch_size = X.size(0) // total_batches

        #looping through batches for training
        for current in range(0, dim_size, total_batches):
            final_value = current + batch_size

            end_val = final_value

            batch_xVals = the_X_vals[current: end_val]
            batch_yVals = the_Y_vals[current: end_val]

           #making our expectations
            expectation = net(batch_xVals)
            loss = loss_fn(expectation, batch_yVals)

            #Optimization for our FNN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    #Everything below is for plotting
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
    plt.title('Vanderpol Graph - Blue is actual and Red is predicted values')
    plt.show()

# Handwritten digit classification using MNIST
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.99)

    # Convert Strings to Ints
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Random sample of test data
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

    # Evaluate model on test data
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
