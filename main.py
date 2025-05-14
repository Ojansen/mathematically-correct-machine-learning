from ops import ReLU, Sigmoid
from loss import Loss
from nn.perceptron import Perceptron

import numpy as np
import time


start = time.time()


# Input vector (2 features)
x = np.array(
    [
        np.random.random(),
        np.random.random(),
        np.random.random(),
        np.random.random(),
        np.random.random(),
    ]
)

# True target value
y_true = np.array([1])

# Initialize weights and bias
W = np.full(5, 1.0)  # Random weights
b = 1  # Random bias

# Learning rate
alpha = 0.01

# Number of training epochs
epochs = 1000

for epoch in range(epochs):
    # Forward Pass
    neuron = Perceptron(W, x, b)
    y_pred = ReLU.forward(neuron.linear())  # Activation function

    # Compute Loss
    loss = Loss(y_true, y_pred).mean_squared_error()

    # Backward Pass (Backpropagation)
    error = y_true - y_pred  # Difference between target and prediction

    # Compute gradients
    dL_dy = -2 * error  # Derivative of MSE loss
    dy_dz = ReLU.backward(y_pred)  # Derivative of sigmoid
    dz_dW = x  # Gradient of z w.r.t. weights
    dz_db = 1  # Gradient of z w.r.t. bias

    # Chain rule application
    grad_W = dL_dy * dy_dz * dz_dW
    grad_b = dL_dy * dy_dz * dz_db

    # Update Weights
    W -= alpha * grad_W
    b -= alpha * grad_b

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: Loss = {loss:.9f}: Prediction {ReLU.forward(Perceptron(W, x, b).linear()):.9f}"
        )

# Final weights
print("Updated Weights:", W)
print("Updated Bias:", b)

end = time.time()

print(f"Time taken to run the code was {end - start} seconds")
