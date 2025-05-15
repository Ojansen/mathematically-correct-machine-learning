import numpy as np


class He:
    @staticmethod
    def initialize(n_layers) -> float:
        return np.random.normal(0, np.sqrt(2 / n_layers))
