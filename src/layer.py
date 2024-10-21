import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, min_value=-0.1, max_value=0.1):
        """
        Initialize a layer in the neural network.

        Parameters:
        - input_size (int): Number of neurons in the previous layer.
        - output_size (int): Number of neurons in the current layer.
        - activation (function): Activation function to be applied to the layer's output.
        - min_value (float): Minimum value for initializing weights.
        - max_value (float): Maximum value for initializing weights.

        The biases are absorbed into the weights by adding an extra input with a constant value (bias term),
        so the weights matrix includes weights for both inputs and biases.
        """
        # Initialize weights with shape (input_size + 1, output_size) to account for bias
        self.weights = np.random.uniform(
            min_value, max_value, (input_size + 1, output_size)
        )
        self.activation = activation  # Activation function for this layer
        self.inputs = None            # Inputs to the layer (including bias term)
        self.grad_weights = np.zeros_like(self.weights)  # Gradient accumulator

    def forward(self, inputs, store_input=False):
        """
        Perform the forward pass for this layer.

        Parameters:
        - inputs (ndarray): Input data or activations from the previous layer.
        - store_input (bool): Whether to store inputs for backpropagation.

        Returns:
        - Activated outputs of this layer.
        """
        # Append bias term to inputs
        inputs_with_bias = np.append(inputs, 1)

        if store_input:
            self.inputs = inputs_with_bias  # Store inputs for backward pass

        # Compute linear combination
        z = np.dot(inputs_with_bias, self.weights)

        # Apply activation function
        output = self.activation(z)
        return output

    def backward(self, delta, activation_prev, inputs_prev):
        """
        Perform the backward pass for this layer.

        Parameters:
        - delta (ndarray): Gradient of the loss with respect to the output of this layer.
        - activation_prev (ndarray): Activation from the previous layer.
        - inputs_prev (ndarray): Inputs to this layer during the forward pass.

        Returns:
        - delta_prev (ndarray): Gradient of the loss with respect to the input of this layer.
        """
        # Compute gradient with respect to z
        derivative = self.activation(np.dot(inputs_prev, self.weights), derivative=True)
        delta *= derivative

        # Accumulate gradients for weights
        inputs_with_bias = np.append(activation_prev, 1)
        self.grad_weights += np.outer(inputs_with_bias, delta)

        # Compute delta for previous layer (excluding bias term)
        delta_prev = np.dot(self.weights[:-1], delta)
        return delta_prev

    def update_weights(self, learning_rate, batch_size, regularization):
        """
        Update the weights of this layer using the accumulated gradients.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        - batch_size (int): Number of samples in the batch.
        - regularization (float): Regularization parameter.
        """
        # Apply regularization
        reg_term = regularization * self.weights
        # Update weights and reset gradients
        self.weights -= learning_rate * ((self.grad_weights / batch_size) + reg_term)
        self.grad_weights.fill(0)  # Reset after update

    def init_gradients(self):
        """
        Initialize gradients to zero before processing a new batch.
        """
        self.grad_weights = np.zeros_like(self.weights)
