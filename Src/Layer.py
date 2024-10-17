import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size + 1, output_size) * 0.1
        self.activation = activation
        self.grad_weights = np.zeros_like(self.weights)  # Gradient accumulator

    def forward(self, inputs):
        self.inputs = np.append(inputs, 1)  # Adding bias as last element
        self.z = np.dot(self.inputs, self.weights)
        return self.activation(self.z)

    def backward(self, grad_output):
        grad_input = grad_output * self.activation(self.z, derivative=True)
        self.grad_weights += np.outer(self.inputs, grad_input)
        return np.dot(self.weights[:-1], grad_input)  # Gradient for previous layer

    def update_weights(self, learning_rate, batch_size):
        # Update weights using accumulated gradients and reset
        self.weights -= learning_rate * (self.grad_weights / batch_size)
        self.grad_weights.fill(0)  # Reset after update
