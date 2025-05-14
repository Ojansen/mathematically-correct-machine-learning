import numpy as np


class ReLU:
    @staticmethod
    def forward(x: float) -> float:
        # return np.maximum(x, 0)
        return float((x + abs(x)) / 2)

    @staticmethod
    def backward(x: float) -> float:
        return float(x > 0)


class Sigmoid:
    @staticmethod
    def forward(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    @staticmethod
    def backward(x: float) -> float:
        return float(x * (1 - x))
