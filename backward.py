import numpy as np
import math
from loss import Loss


def xlogy(x: np.ndarray, y: np.ndarray):
    return x * np.log(y)


def binary_crossentropy(target, pred) -> float:
    return -target * math.log(pred) - (1 - target) * math.log((1 - pred))


# https://www.jasonosajima.com/backprop
z3 = 1.1
y = 1.0
loss = binary_crossentropy(y, z3)
print(loss)
# [6.40806422 6.40806422 6.40806422 6.40806422]
# [25.73513698 25.73513698]
# [1.]
