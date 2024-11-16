import ssl
import time
import torch
import numpy as np
import torch.nn as nn
from FNN import FeedforwardNeuralNetwork
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
import random

# Store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    x = np.clip(x, -100, 100) # Prevent weights from overflowing
    x += + 1e-10 # Prevent divide by 0
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

# Custom loss function with optional derivative
def NNN_loss(y_pred, y_true, derivative=False):
    if derivative:
        grad = np.copy(y_pred)
        ###if NNN_loss_counter == 0:
        ###    print("y_pred.shape: ", y_pred.shape)
        ###    print("y_true value: ", y_true)
        grad[range(len(y_pred)), y_true] -= 1
        return grad / len(y_pred)  # Gradient with respect to the input

    return np.mean(-y_pred[range(len(y_pred)), y_true])

### Flags for different datasets

run_digits = False
run_covtype = True

###

### Flags for different algorithms

xavier = True
# (set either nesterov_ or adam_ to true or neither for vanilla GD)
nesterov_ = True
adam_ = False

####
if adam_:
    LEARNING_RATE = 0.001
elif nesterov_:
    LEARNING_RATE = .1
else:
    LEARNING_RATE = 1

if run_covtype:
    EPOCHS = 10
    layer_sizes = [54, 128, 128, 128, 7]
    # Fetch the Forest Covertypes dataset
    forest = fetch_covtype()

    # Save feature data to X and target values to y
    X = forest.data  # Contains the feature data
    y = forest.target  # Contains the target labels
    y = y - 1
    
    # Normalization (from features with varying ranges to a set range of [0,1] for all features.
    data_min = np.min(X, axis=0)  # Minimum value along each feature
    data_max = np.max(X, axis=0)  # Maximum value along each feature
    X = (X - data_min) / (data_max - data_min)
    
    # Split data into train partition and test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.95)
    _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=0, test_size=0.1)
    # Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron
    nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid, sigmoid,sigmoid,sigmoid], xavier=xavier, nesterov_=nesterov_, adam_=adam_)

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
    correct_counts = np.zeros(7)
    incorrect_counts = np.zeros(7)

    for i in range(len(y_test)):
        y_pred = nn.forward(X_test[i])
        if np.random.randint(1, 1000) == 1:
            print(y_pred)
        if np.argmax(y_pred)  ==  y_test[i]:
            correct_counts[y_test[i]] += 1
        else:
            incorrect_counts[y_test[i]] += 1

    print("correct: ", np.sum(correct_counts), "incorrect: ", np.sum(incorrect_counts))
    print("accuracy: ", np.sum(correct_counts) / (np.sum(correct_counts) + np.sum(incorrect_counts))) 

    covtype = np.arange(7)

    # Plotting correct vs incorrect classifications for each digit
    plt.figure(figsize=(7, 6)) 
    plt.bar(covtype, correct_counts, label="Correct", color="blue")
    plt.bar(covtype, incorrect_counts, bottom=correct_counts, label="Incorrect", color="red")

    # Labeling the plot
    plt.xlabel("Covtype")
    plt.ylabel("Number of Classifications")
    plt.title(f"Correct vs Incorrect | Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}, T:{training_duration} min,Correct: {((np.sum(correct_counts)/len(y_test)) * 100):.2f}%")
    plt.xticks(covtype)
    plt.legend()

    # Show plot
    plt.show()


# Handwritten digit classification using MNIST
if(run_digits):
    layer_sizes = [784, 128, 128, 10]
    EPOCHS = 100
    TEST_SIZE = 10000

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
    nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid,sigmoid,sigmoid], xavier=xavier, nesterov_=nesterov_, adam_=adam_)

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

