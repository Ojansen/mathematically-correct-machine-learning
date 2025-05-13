from ops import relu, sigmoid
from nn.perceptron import Perceptron

print(relu(-1))
print(relu(2))

print(sigmoid(10))
print(sigmoid(-10))

x = [1.0, 2.0]
w = [0.5, -1.0]
b = 0.1

neuron = Perceptron(w, x, b)
print(neuron.linear())
print(neuron.sigmoid())
