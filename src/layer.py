import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation):
        # Initialize weights with shape (input_size + 1, output_size) to account for bias
        self.weights = np.random.randn(input_size + 1, output_size) * 0.1
        self.activation = activation
        self.grad_weights = np.zeros_like(self.weights)  # Gradient accumulator
        self.inputs = None  # Initialize inputs as None, to be set during forward pass

    def forward(self, inputs):
        # Ensure inputs are 2D, with shape (batch_size, input_size)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)  # Convert to 2D array if it's a single sample

        # Store inputs for use in the backward pass
        self.inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)  # Append bias term
        self.z = np.dot(self.inputs, self.weights)
        return self.activation(self.z)

    def backward(self, grad_output):
        grad_input = grad_output * self.activation(self.z, derivative=True)
        # Accumulate gradients for weight updates
        self.grad_weights += np.outer(self.inputs.T, grad_input)  # Use stored inputs
        return np.dot(grad_input, self.weights[:-1].T)  # Gradient for previous layer (exclude bias)

    def update_weights(self, learning_rate, batch_size):
        # Update weights using accumulated gradients and reset
        self.weights -= learning_rate * (self.grad_weights / batch_size)
        self.grad_weights.fill(0)  # Reset after update