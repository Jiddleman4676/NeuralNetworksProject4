from layer import Layer
import numpy as np

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

    def backward(self, y_pred, y_true, loss_derivative):
        # Gradient of the loss function
        grad = loss_derivative(y_pred, y_true)
        # Backward pass through each layer
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def gd(self, learning_rate, batch_size):
        for layer in self.layers:
            # Update weights using accumulated gradients
            layer.weights -= learning_rate * (layer.grad_weights / batch_size)

            #Reset gradient
            layer.grad_weights.fill(0)

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
                    self.backward(y_pred, y_true, loss_derivative)

                # After processing batch, perform gradient descent
                self.gd(learning_rate, batch_size)