import numpy as np
from layer import Layer

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, num_epochs=1000, regularization=0):
        """
        Initialize the feedforward neural network.

        Parameters:
        - layer_sizes (list): List specifying the number of neurons in each layer.
        - activations (list): List of activation functions for each layer (excluding input layer).
        - learning_rate (float): Learning rate for weight updates.
        - num_epochs (int): Number of epochs for training.
        - regularization (float): Regularization parameter (lambda) for weight decay.
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.regularization = regularization

        # Initialize layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                min_value=-0.1,
                max_value=0.1
            )
            self.layers.append(layer)

    def forward(self, input_vector):
        """
        Perform the forward pass through the network for a single input vector.

        Parameters:
        - input_vector (ndarray): Input data, shape (input_size,).

        Returns:
        - output_vector (ndarray): Output of the network.
        """
        activation = input_vector
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def train(self, inputs, outputs, loss_function, loss_derivative, batch_size=32):
        """
        Train the neural network using mini-batch gradient descent.

        Parameters:
        - inputs (ndarray): Training input data, shape (num_samples, input_size).
        - outputs (ndarray): Training output data, shape (num_samples, output_size).
        - loss_function (function): Loss function.
        - loss_derivative (function): Derivative of loss function w.r.t. network output.
        - batch_size (int): Size of each mini-batch.
        """
        num_samples = inputs.shape[0]
        for epoch in range(self.num_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(num_samples)
            inputs_shuffled = inputs[indices]
            outputs_shuffled = outputs[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = inputs_shuffled[start_idx:end_idx]
                batch_outputs = outputs_shuffled[start_idx:end_idx]
                batch_size_actual = end_idx - start_idx

                # Forward and backward passes for the batch
                self._train_batch(batch_inputs, batch_outputs, loss_derivative, batch_size_actual)

            # Optional: Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                predictions = self.predict(inputs)
                loss = loss_function(predictions, outputs)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss}")

    def _train_batch(self, batch_inputs, batch_outputs, loss_derivative, batch_size):
        """
        Train the network on a single batch of data.

        Parameters:
        - batch_inputs (ndarray): Batch input data.
        - batch_outputs (ndarray): Batch output data.
        - loss_derivative (function): Derivative of loss function.
        - batch_size (int): Number of samples in the batch.
        """
        # Initialize gradients for weights
        for layer in self.layers:
            layer.init_gradients()

        for x, y_true in zip(batch_inputs, batch_outputs):
            # Forward pass
            activations = [x]
            inputs = []
            for layer in self.layers:
                x = layer.forward(x, store_input=True)
                activations.append(x)
                inputs.append(layer.inputs)

            # Backward pass
            delta = loss_derivative(activations[-1], y_true)
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                delta = layer.backward(delta, activations[i], inputs[i])

        # Update weights after processing the batch
        for layer in self.layers:
            layer.update_weights(
                learning_rate=self.learning_rate,
                batch_size=batch_size,
                regularization=self.regularization
            )

    def predict(self, inputs):
        """
        Predict the outputs for given input data.

        Parameters:
        - inputs (ndarray): Input data, shape (num_samples, input_size) or (input_size,).

        Returns:
        - predictions (ndarray): Predicted outputs.
        """
        if inputs.ndim == 1:
            return self.forward(inputs)
        else:
            predictions = []
            for x in inputs:
                predictions.append(self.forward(x))
            return np.array(predictions)
