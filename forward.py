import numpy as np
from ops import ReLU, Sigmoid

# https://www.jasonosajima.com/forwardprop.html
# Input features
X = np.array([2.23123, 1.2313, 4.321])
# Weight and bias layers
W1 = np.full((4, 3), 0.7)
b1 = np.full(4, np.random.random())

W2 = np.full((2, 4), 1)
b2 = np.full(2, np.random.random())

W3 = np.full((1, 2), 1.41)
b3 = np.random.random()

# input -> Hidden layer 1
z1 = X @ W1.transpose() + b1
a1 = ReLU.forward(z1)
print(a1)

# Hidden layer 1 -> Hidden layer 2
z2 = a1 @ W2.transpose() + b2
a2 = ReLU.forward(z2)
print(a2)

# Hidden layer 1 -> Output layer
z3 = a2 @ W3.transpose() + b3
a3 = Sigmoid.forward(z3)
print(a3)
