import numpy as np


class Dense:
    def __init__(self, activation, initializer):
        self.activation = activation
        self.initializer = initializer
