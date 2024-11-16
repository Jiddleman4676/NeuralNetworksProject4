import numpy as np
from layer import Layer

class FeedforwardNeuralNetwork:
    """
    Initializes the network layers with specified sizes and activation functions.
    Each layer connects to the next, forming a feedforward structure.
    """
    def __init__(self, layer_sizes, activations, xavier, nesterov_, adam_):
        # Initialize the layers of the network
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i], xavier))
        self.nesterov_ = nesterov_
        self.adam_ = adam_

    def forward(self, x):
        """
        Performs a forward pass through the network.
        Each layer processes the input sequentially, returning the final output.
        """
        # Forward pass through each layer
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true, loss_derivative):
        """
        Performs a backward pass through the network.
        Calculates gradients starting from the output layer and propagates them back.
        """
        grad = loss_derivative(y_pred, y_true) # Gradient of the loss function
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def gd(self, learning_rate, batch_size):
        """
        Performs gradient descent to update weights for each layer.
        Uses accumulated gradients from the backward pass to adjust weights.
        """
        for layer in self.layers:
            layer.weights -= learning_rate * (layer.grad_weights / batch_size)
            layer.grad_weights.fill(0)

    def adam(self, learning_rate, batch_size, b1=0.9, b2=0.999, epsilon=1e-8):
        """
        Performs Adam optimization to update weights for each layer.
        :param learning_rate: Learning rate for updates.
        :param batch_size: Number of samples in a batch.
        :param b1: Decay rate for the first moment.
        :param b2: Decay rate for the second moment.
        :param epsilon: Small constant for numerical stability.
        """
        for layer in self.layers:
            layer.t += 1
            layer.m = b1 * layer.m + (1 - b1) * (layer.grad_weights / batch_size)
            layer.v = b2 * layer.v + (1 - b2) * (layer.grad_weights / batch_size) ** 2

            # Correct bias
            m_h = layer.m / (1 - b1 ** layer.t)
            v_h = layer.v / (1 - b2 ** layer.t)

            # Update weights
            layer.weights -= learning_rate * m_h / (np.sqrt(v_h) + epsilon)
            layer.grad_weights.fill(0)

    def nesterov(self, learning_rate, batch_size, momentum=0.85):
        """
        Applies Nesterov momentum-based updates to each layer's weights.
        """
        for layer in self.layers:
            if not hasattr(layer, 'v'):
                layer.v = np.zeros_like(layer.weights)

            # Lookahead calculation with custom handling of previous velocity
            v_prev = np.copy(layer.v)  # Store the previous velocity
            layer.v = momentum * layer.v - learning_rate * (layer.grad_weights / batch_size)

            # Update weights with Nesterov correction based on prior and current velocities
            layer.weights += momentum * v_prev - learning_rate * (layer.grad_weights / batch_size)
            layer.grad_weights.fill(0)  # Reset gradient accumulator

    def train(self, X, Y, loss_function, loss_derivative, epochs=1000, learning_rate=0.01, batch_size=32):
        """
        Trains the neural network over multiple epochs.
        Each epoch processes the dataset in mini-batches for efficient gradient descent.
        """
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, " out of ", epochs)
            indices = np.random.permutation(len(X)) # Shuffle data indices for each epoch
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X, batch_Y = X[batch_indices], Y[batch_indices]

                # Forward and backward pass for each sample in the batch
                for x, y_true in zip(batch_X, batch_Y):
                    y_pred = self.forward(x)
                    self.backward(y_pred, y_true, loss_derivative)

                # After processing batch, choose to use adams algorithm or gd

                if self.adam_:
                    self.adam(learning_rate, batch_size)
                elif self.nesterov_:
                    self.nesterov(learning_rate, batch_size)
                else:
                    self.gd(learning_rate, batch_size)
