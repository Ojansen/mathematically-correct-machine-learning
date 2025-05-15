from ops import ReLU, Sigmoid, Softmax
from loss import Loss

import numpy as np
import time
import nn.models as models
from nn.perceptron import Perceptron

from matplotlib import pyplot as plt

y = np.array([0, 1, 0, 1, 1, 0])

y_pred = np.array([0.3, 0.8, 0.4, 0.7, 0.9, 0.2])

start = time.time()
# Input vector (2 features)
X = np.array(
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

# if __name__ == "__main__":
#     model = models.MLPRegressor(alpha=0.01)

#     model.fit(X, y, y_true, epochs=1000)

# model.predict()
