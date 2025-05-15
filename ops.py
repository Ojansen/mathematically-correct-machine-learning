import numpy as np


class ReLU:
    @staticmethod
    def forward(x: float) -> float:
        # return np.maximum(x, 0)
        return (x + abs(x)) / 2

    @staticmethod
    def backward(x: float) -> float:
        return float(x > 0)


class Sigmoid:
    @staticmethod
    def forward(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def backward(x: float) -> float:
        return float(x * (1 - x))


class Softmax:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        pass
