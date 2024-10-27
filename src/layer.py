import numpy as np

def dot_product(a, b):
    """
    Manually computes the dot product between two matrices a and b.
    """
    result = []
    for i in range(len(a)):
        row_result = []
        for j in range(len(b[0])):
            sum_value = 0
            for k in range(len(b)):
                sum_value += a[i][k] * b[k][j]
            row_result.append(sum_value)
        result.append(row_result)
    return result

class Layer:

    #input_size - number of neurons in the previous layer
    #output_size - number of neurons in the next layer

    def __init__(self, input_size, output_size, activation):
        """
        Initializes the layer by setting the weights, activation function, and
        gradient accumulator for the weights. The weights include an additional
        term for bias.
        """
        self.z = None
        #set the number of weights + 1 to absorb the bias
        self.weights = np.random.randn(input_size + 1, output_size) * 0.1
        self.activation = activation
        #will store the accumulated grads for the weights
        self.grad_weights = np.zeros_like(self.weights)
        self.inputs = None

    def forward(self, inputs):
        """
        Performs the forward pass through the layer. Appends a bias term to the inputs,
        computes the dot product of the inputs and weights, applies the activation function,
        and returns the result.
        """
        #turn row vector into column vector
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, len(inputs))

        # Add bias term by appending 1s to the inputs
        bias_term = np.ones((inputs.shape[0], 1))  # Create a column of 1s
        self.inputs = np.concatenate((inputs, bias_term), axis = 1)  # Append bias term

        # Calculate dot product
        self.z = np.dot(self.inputs, self.weights)

        # Convert list to numpy array for activation function
        self.z = np.array(self.z)

        # Return the activated output
        return self.activation(self.z)

    def backward(self, grad_output):
        """
        First computes the gradients of the inputs
        Then computes the gradient of the weights (grad_weights).
        Then it is accumulated in grad_weights for the update after the mini batch is completed.
        """

        # grad of current layers output with respect to its input
        grad_input = grad_output * self.activation(self.z, derivative=True)  # Shape: (batch_size, output_size)

        # Compute the gradient for weights (grad_weights)
        # ie how do the weights need to change (will be used in update weights)
        grad_weights_update = np.dot(self.inputs.T, grad_input)
        self.grad_weights += grad_weights_update  # Accumulate grads over mini-batch

        # Compute gradient for the previous layer (excluding the bias term)
        grad_previous_layer = np.dot(grad_input, self.weights[:-1].T)  # Shape: (batch_size, input_size)

        # Return gradient to propagate to the previous layer
        return grad_previous_layer

    def update_weights(self, learning_rate, batch_size):
        """
        After the mini batch is complete, all the weights are updated
        Then the acc is reset
        """
        self.weights -= learning_rate * (self.grad_weights / batch_size)

        # Reset the accumulator
        self.grad_weights.fill(0)
