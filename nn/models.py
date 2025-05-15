import numpy as np
from loss import Loss
from ops import ReLU
from nn.optim import SGD
from nn.perceptron import Perceptron
from nn.utils import He


class MLPRegressor:
    def __init__(
        self,
        hidden_layers=(100,),
        activation=ReLU(),
        loss=Loss,
        alpha=0.01,
        optimizer=SGD(),
        initializer=He,
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.loss = loss
        self.optimizer = optimizer
        self.initializer = initializer

    def forward(self):
        pass

    def backward(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, y_target: np.ndarray, epochs: int):
        W = np.full(X.size, 1.0)
        b = 1

        for epoch in range(epochs):
            # Forward Pass
            neuron = Perceptron(W, X, b)
            y_pred = self.activation.forward(neuron.linear())  # Activation function

            # Compute Loss
            loss = self.loss(y_target, y_pred).mean_squared_error()

            # Backward Pass (Backpropagation)
            error = y_target - y_pred  # Difference between target and prediction

            # Compute gradients
            dL_dy = -2 * error  # Derivative of MSE loss
            dy_dz = self.activation.backward(y_pred)  # Derivative of sigmoid
            dz_dW = X  # Gradient of z w.r.t. weights
            dz_db = 1  # Gradient of z w.r.t. bias

            # Chain rule application
            grad_W = dL_dy * dy_dz * dz_dW
            grad_b = dL_dy * dy_dz * dz_db

            # Update Weights
            W -= self.alpha * grad_W
            b -= self.alpha * grad_b

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: Loss = {loss:.9f}: Prediction {self.activation.forward(Perceptron(W, X, b).linear()):.9f}"
                )

    def predict(self, X: np.ndarray):
        pass
