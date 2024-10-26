from layer import Layer

import numpy as np


def gd(y_pred, y_true, loss_derivative):
    return loss_derivative(y_pred, y_true)


class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        # Initialize the layers of the network
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, x):
        # Forward pass through each layer
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true, loss_derivative, learning_rate):
        # Compute the gradient of the loss with respect to the network's output
        grad = gd(y_pred, y_true, loss_derivative)
        # Backward pass through each layer
        for layer in reversed(self.layers):
            # Propagate the gradient backwards through the layer
            grad = layer.backward(grad, learning_rate)

    def train(self, X, Y, loss_function, loss_derivative, epochs=1000, learning_rate=0.01, batch_size=32):
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, " out of ", epochs)
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X, batch_Y = X[batch_indices], Y[batch_indices]

                # Forward and backward pass for each sample in the batch
                for x, y_true in zip(batch_X, batch_Y):
                    y_pred = self.forward(x)
                    grad = loss_derivative(y_pred, y_true)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad)  # Accumulate gradients in each layer

                # After batch, update weights with accumulated gradients
                for layer in self.layers:
                    layer.update_weights(learning_rate, batch_size)