import ops


def test_relu():
    assert ops.relu(1) == 1
    assert ops.relu(-1) == 0
