import ops


def test_relu():
    relu = ops.ReLU()
    assert relu.forward(3) == 3.0
    assert relu.forward(-3) == 0
    assert relu.backward(-1) == 0
    assert relu.backward(3) == 1


def test_sigmoid():
    sigmoid = ops.Sigmoid()
    assert sigmoid.forward(1) > 0.5
    assert sigmoid.forward(-1) < 0.5
