import numpy as np

class Layer:

    def __init__(self, input_size, output_size, activation, xavier):
        """
        Initializes the layer with Xavier initialization for weights,
        activation function, and gradient accumulator for weights.
        """
        self.z = None
        self.activation = activation

        
        # Customized Xavier initialization, taking into account bias 
        if xavier:
            scale = np.sqrt(6 / (input_size + output_size + 1))
            self.weights = np.random.uniform(-scale, scale, (input_size + 1, output_size))
        else:
            #set the number of weights + 1 to absorb the bias
            self.weights = np.random.randn(input_size + 1, output_size) * 0.1
        # Accumulator for gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.inputs = None

        # Variables for Nesterov and Adam momentum
        self.m = np.zeros_like(self.weights)  # First moment vector
        self.v = np.zeros_like(self.weights)  # Second moment vector
        self.b1 = 0.9  # Decay rate for first moment
        self.b2 = 0.999  # Decay rate for second moment
        self.t = 0  # Time step

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
        grad_input = grad_output * self.activation(self.z, derivative=True)  # Shape: (batch_num, output_size)

        # Compute the gradient for weights (grad_weights)
        # ie how do the weights need to change (will be used in update weights)
        grad_weights_update = np.dot(self.inputs.T, grad_input)
        self.grad_weights += grad_weights_update  # Accumulate grads over mini-batch

        # Compute gradient for the previous layer (excluding the bias term)
        grad_previous_layer = np.dot(grad_input, self.weights[:-1].T)  # Shape: (batch_num, input_size)

        # Return gradient to propagate to the previous layer
        return grad_previous_layer
