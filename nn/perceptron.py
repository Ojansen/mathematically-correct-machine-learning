import ops


class Perceptron:
    def __init__(self, w: list, x: list, b: int):
        self.dot = sum([i * j for i, j in zip(w, x)]) + b

    def linear(self) -> float:
        return self.dot

    def sigmoid(self) -> float:
        return ops.sigmoid(self.dot)
