import loss
import numpy as np

pred = np.array([1, 1, 0, 0])
target = np.array([1, 1, 1, 1])
instance = loss.Loss(pred, target)


def test_loss():
    assert instance.mean_squared_error() == 0.5
    assert instance.mean_absolute_error() == 0.5
    assert instance.root_mean_squared_error() < 0.5
