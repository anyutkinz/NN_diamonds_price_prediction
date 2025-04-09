import pandas as pd
import numpy as np
import os

class DiamondPricePredictor:
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, learning_rate=0.001):
        """
        Initialize the DiamondPricePredictor class with architecture and optimizer parameters.

        Args:
            input_dim (int): Number of input features.
            hidden_layer_sizes (list): List of neuron counts for each hidden layer.
            output_dim (int): Number of neurons in the output layer.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.

        Attributes:
            weights (list): List of weight matrices for each layer.
            biases (list): List of bias vectors for each layer.
            layer_outputs (list): List to store outputs of each hidden layer during forward propagation.
            m_weights, m_biases, v_weights, v_biases (list): Adam optimizer parameters for momentum and variance.
            beta1, beta2, epsilon (float): Adam optimizer hyperparameters.
            t (int): Time step for Adam optimizer updates.
        """
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        self.layer_outputs = []

        self._initialize_weights_and_biases()
        
        # Initialize Adam optimizer parameters (momentum and variance terms)
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.beta1 = 0.9  # Momentum decay rate
        self.beta2 = 0.999  # Variance decay rate
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.t = 0 # Time step for Adam optimizer

    def _initialize_weights_and_biases(self):
        """
        Initialize weights and biases for each layer.
        Weights are randomly initialized, and biases start as zeros.

        Attributes:
            weights (list): Each weight matrix has dimensions [layer_in, layer_out].
            biases (list): Each bias vector has dimensions [1, layer_out].
        """
        layers = [self.input_dim] + self.hidden_layer_sizes + [self.output_dim]
        for i in range(len(layers) - 1): # Loop through pairs of layers
            
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.01 # Initialize weights with small random values
            bias = np.zeros((1, layers[i + 1])) # Initialize biases as zeros
            self.weights.append(weight)
            self.biases.append(bias)
            
    def _forward_propagation(self, X):
        """
        Perform forward propagation through the network with usage of ReLU activation function.

        Args:
            X (np.array): Input feature matrix of shape [n_samples, input_dim].

        Returns:
            np.array: Output predictions (raw scores) from the final layer.
        """
        self.layer_outputs = [] # Reset layer outputs for each forward pass
        input_data = X
        
        for i in range(len(self.weights)): # Loop through all layers
            Z = np.dot(input_data, self.weights[i]) + self.biases[i] # Linear transformation
            # Apply activation functions only for hidden layers
            if i < len(self.weights) - 1:
                input_data = np.maximum(0, Z) # ReLU activation function
            else:
                input_data = Z # Linear output for the final layer
            
            if i < len(self.weights) - 1: # Exclude the output layer
                self.layer_outputs.append(input_data) # Store outputs for backpropagation
            
        return input_data
    
    def _compute_cost(self, y_hat, y):
        """
        Compute the cost using Mean Squared Error (MSE).

        Args:
            y_hat (np.array): Predictions from the model.
            y (np.array): Ground truth labels.

        Returns:
            float: Mean Squared Error between predictions and ground truth.
        """
        n = y.shape[0] # Number of samples
        cost = np.sum((y_hat - y) ** 2) / n # Average squared difference
        return cost
    
    def _backward_propagation(self, X, y_hat, y):
        """
        Perform backward propagation to compute gradients.

        Args:
            X (np.array): Input feature matrix.
            y_hat (np.array): Predicted values.
            y (np.array): Ground truth values.

        Returns:
            Tuple[list, list]: Gradients for weights and biases.
        """
        n = y.shape[0] # Number of samples
        
        # Initialize gradients for weights and biases
        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)
        
        # Compute the gradient of the cost with respect to the output layer
        d_z = (2 / n) * (y_hat - y) # MSE gradient
        
        for i in reversed(range(len(self.weights))): # Backpropagate through layers
            d_weights[i] = np.dot(self.layer_outputs[i - 1].T if i > 0 else X.T, d_z) 
            d_biases[i] = np.sum(d_z, axis=0, keepdims=True)
            
            if i > 0: # Backpropagate through ReLU activation function
                relu_gradient = self.layer_outputs[i - 1] > 0
                d_z = np.dot(d_z, self.weights[i].T) * relu_gradient
                
        return d_weights, d_biases
    
    def _update_parameters(self, d_weights, d_biases):
        """
        Update weights and biases using Adam optimizer.

        Args:
            d_weights (list): Gradients for weights.
            d_biases (list): Gradients for biases.
        """
        self.t += 1 # Increment time step 
        
        for i in range(len(self.weights)):
            # Compute momentum terms
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * d_weights[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * d_biases[i]
            
            # Compute variance terms
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (d_weights[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (d_biases[i] ** 2)
            
            # Bias correction
            m_weights_corrected = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_biases_corrected = self.m_biases[i] / (1 - self.beta1 ** self.t)
            
            v_weights_corrected = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon) 
    
    def fit(self, X_train, y_train, epochs, batch_size):
        """
        Train the model using mini-batch gradient descent.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            epochs (int): Number of iterations over the entire dataset.
            batch_size (int): Number of samples processed per batch.
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs): # Loop through epochs
            indices = np.random.permutation(n_samples) # Shuffle the data
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_cost = 0
            
            for i in range(0, n_samples, batch_size): # Process data in batches
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                # Forward propagation
                y_hat = self._forward_propagation(X_batch)
                
                # Compute cost
                cost = self._compute_cost(y_hat, y_batch)
                epoch_cost += cost
                
                # Backward propagation
                d_weights, d_biases = self._backward_propagation(X_batch, y_hat, y_batch)
                
                # Update parameters
                self._update_parameters(d_weights, d_biases)
            
            epoch_cost /= (n_samples // batch_size) # Average cost across batches
            print(f"Epoch {epoch + 1}/{epochs}, Cost: {epoch_cost:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Args:
            X_test (np.array): Test features.

        Returns:
            np.array: Predictions for test data.
        """
        y_hat = self._forward_propagation(X_test)
        return y_hat
    
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Mean Squared Error (MSE).
        
        Args:
            X_test (np.array): Test feature matrix.
            y_test (np.array): Ground truth test labels.
        
        Returns:
            float: The computed Mean Squared Error (MSE) for predictions on test data.
        """
        y_hat = self.predict(X_test)
        mse = self._compute_cost(y_hat, y_test)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse
    
    def save_model(self, file_path):
        """
        Save the model parameters (weights and biases) to a file.
        
        Args:
            file_path (str): Path to save the model file.
        """
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_parameters = {
            'weights': self.weights,
            'biases': self.biases
        }
        np.save(file_path, model_parameters)
        print(f"Model saved to {file_path}")
                
                