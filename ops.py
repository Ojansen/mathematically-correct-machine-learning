import math


def relu(x) -> int:
    return (x + abs(x)) / 2


def sigmoid(x) -> int:
    return 1 / (1 + math.exp(-x))
