import numpy as np


class Loss:
    def __init__(self, pred: np.ndarray, target: np.ndarray):
        self.pred = pred
        self.target = target

    def mean_squared_error(self) -> float:
        return float(np.mean((self.pred - self.target) ** 2))

    def mean_absolute_error(self) -> float:
        return float(np.sum(abs((self.pred - self.target))) / self.pred.size)

    def root_mean_squared_error(self) -> float:
        return float(np.sqrt(np.mean((self.pred - self.target) ** 2) / self.pred.size))

    def xlogy(self, x: np.ndarray, y: np.ndarray):
        return x * np.log(y)

    def binary_crossentropy(self) -> float:
        return (
            np.sum(
                -(
                    self.xlogy(self.target, self.pred)
                    + self.xlogy(1 - self.target, 1 - self.pred)
                )
            )
            / self.pred.size
        )
