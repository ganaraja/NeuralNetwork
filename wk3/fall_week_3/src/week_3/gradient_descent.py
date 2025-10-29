import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

class GradientDescentClassification:
    def __init__(self, X1: np.ndarray, X2: np.ndarray, Y: np.ndarray, learning_rate: float = 1e-4, epochs: int = 300):
        """
        Initializes the Gradient Descent optimizer with data and parameters.
        
        :param X1: First feature array.
        :param X2: Second feature array.
        :param Y: Target array.
        :param learning_rate: Learning rate for gradient descent.
        :param epochs: Number of iterations for optimization.
        """
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.α = learning_rate
        self.epochs = epochs
        self.b = 2.0
        self.w_1 = 2.0
        self.w_2 = 2.0
    
    def compute_gradient_descent_pytorch(self, α: float = None, epochs: int = None) -> pd.DataFrame:
        """
        Computes gradient descent using PyTorch and returns the intermediate values.
        
        :return: DataFrame containing epoch, weights, bias, and loss values.
        """
        if not α:
            α = self.α
        if not epochs:
            epochs = self.epochs
        X1 = torch.tensor(self.X1, dtype=torch.float32)
        X2 = torch.tensor(self.X2, dtype=torch.float32)
        Y = torch.tensor(self.Y, dtype=torch.float32)
        
        # Initialize parameters with gradient tracking
        b = torch.tensor(self.b, dtype=torch.float32, requires_grad=True)
        w_1 = torch.tensor(self.w_1, dtype=torch.float32, requires_grad=True)
        w_2 = torch.tensor(self.w_2, dtype=torch.float32, requires_grad=True)
        
        intermediates = pd.DataFrame(columns=['epoch', 'b', 'w_1', 'w_2', 'loss'])
        
        for epoch in range(epochs + 1):
            # Compute linear combination and apply sigmoid activation
            z = w_1 * X1 + w_2 * X2 + b
            y_pred = torch.sigmoid(z)
            
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(y_pred, Y)
            
            # Compute gradients
            loss.backward()
            
            with torch.no_grad():
                # Update parameters using gradient descent
                b -= α * b.grad
                w_1 -= α * w_1.grad
                w_2 -= α * w_2.grad
                
                # Zero the gradients after updating
                b.grad.zero_()
                w_1.grad.zero_()
                w_2.grad.zero_()
            
            # Store intermediate results
            intermediates.loc[epoch] = [epoch, b.item(), w_1.item(), w_2.item(), loss.item()]
        
        return intermediates
    
    def compute_gradient_descent_numpy(self) -> pd.DataFrame:
        """
        Computes gradient descent using NumPy and returns the intermediate values.
        
        :return: DataFrame containing epoch, weights, bias, and loss values.
        """
        b, w_1, w_2 = self.b, self.w_1, self.w_2
        intermediates = pd.DataFrame(columns=['epoch', 'b', 'w_1', 'w_2', 'loss'])
        
        def sigma(z: float) -> float:
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-z))
        
        for epoch in range(self.epochs + 1):
            # Compute gradients
            db = -np.sum((self.Y - sigma(w_1 * self.X1 + w_2 * self.X2 + b)))
            dw_1 = -np.sum((self.Y - sigma(w_1 * self.X1 + w_2 * self.X2 + b)) * self.X1)
            dw_2 = -np.sum((self.Y - sigma(w_1 * self.X1 + w_2 * self.X2 + b)) * self.X2)
            
            # Update parameters
            b -= self.α * db
            w_1 -= self.α * dw_1
            w_2 -= self.α * dw_2
            
            # Compute loss using binary cross-entropy
            loss = -np.sum(self.Y * np.log(sigma(w_1 * self.X1 + w_2 * self.X2 + b)) + 
                           (1 - self.Y) * np.log(1 - sigma(w_1 * self.X1 + w_2 * self.X2 + b)))
            
            # Store intermediate results
            intermediates.loc[epoch] = [epoch, b, w_1, w_2, loss]
        
        return intermediates


class GradientDescentRegression:
    def __init__(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 1e-4, epochs: int = 200, initial_weights: list = [4.0 , 4.0]):
        """
        Initializes the Gradient Descent optimizer for Linear Regression.
        
        :param X: Feature array.
        :param Y: Target array.
        :param learning_rate: Learning rate for gradient descent.
        :param epochs: Number of iterations for optimization.
        """
        self.X = X
        self.Y = Y
        self.α = learning_rate
        self.epochs = epochs
        self.β_0 = initial_weights[0]
        self.β_1 = initial_weights[1]
    
    def compute_gradient_descent_numpy(self) -> pd.DataFrame:
        """
        Computes gradient descent for Linear Regression using NumPy.
        
        :return: DataFrame containing epoch, weights, and loss values.
        """
        β_0, β_1 = self.β_0, self.β_1
        intermediates = pd.DataFrame(columns=['epoch', 'β_0', 'β_1', 'loss'])
        
        for epoch in range(self.epochs + 1):
            # Compute gradients
            dβ_0 = -np.sum(self.Y - β_0 - β_1 * self.X)
            dβ_1 = -np.sum(self.X * (self.Y - β_0 - β_1 * self.X))
            
            # Update parameters
            β_0 -= self.α * dβ_0
            β_1 -= self.α * dβ_1
            
            # Compute loss (Mean Squared Error)
            loss = np.sum((self.Y - β_0 - β_1 * self.X) ** 2)
            
            # Store intermediate results
            intermediates.loc[epoch] = [epoch, β_0, β_1, loss]
        
        return intermediates

    
    def compute_gradient_descent_pytorch(self) -> pd.DataFrame:
        """
        Computes gradient descent for Linear Regression using PyTorch.
        
        :return: DataFrame containing epoch, weights, and loss values.
        """
        import torch

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.float32)

        # Initialize parameters as tensors with gradients
        β_0 = torch.tensor(self.β_0, dtype=torch.float32, requires_grad=True)
        β_1 = torch.tensor(self.β_1, dtype=torch.float32, requires_grad=True)

        intermediates = pd.DataFrame(columns=['epoch', 'β_0', 'β_1', 'loss'])

        for epoch in range(self.epochs + 1):
            # Compute predictions
            Y_pred = β_0 + β_1 * X_tensor

            # Compute Mean Squared Error (MSE) loss
            loss = torch.sum((Y_tensor - Y_pred) ** 2)

            # Backpropagation
            loss.backward()

            # Update parameters
            with torch.no_grad():
                β_0 -= self.α * β_0.grad
                β_1 -= self.α * β_1.grad

            # Zero gradients for next iteration
            β_0.grad.zero_()
            β_1.grad.zero_()

            # Store intermediate values
            intermediates.loc[epoch] = [epoch, β_0.item(), β_1.item(), loss.item()]

        return intermediates