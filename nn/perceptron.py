from ops import Sigmoid
import numpy as np


class Perceptron:
    def __init__(self, w: np.ndarray, x: np.ndarray, b: float):
        self.dot = np.dot(w, x) + b

    def linear(self) -> float:
        return self.dot

    def sigmoid(self) -> float:
        return Sigmoid.forward(self.dot)
