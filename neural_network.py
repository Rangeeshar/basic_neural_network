import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, w1, w2, bias):
        self.weights = np.array([w1, w2])
        self.bias = bias
        # The Neuron class here is from the previous section
        # TODO: Dynamically create neurons and hidden layers.
        self.h1 = Neuron(self.weights, self.bias)
        self.h2 = Neuron(self.weights, self.bias)
        self.o1 = Neuron(self.weights, self.bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1

if __name__=="__main__":
    network = NeuralNetwork(0, 1, 0)
    x = np.array([2, 3])
    print(network.feedforward(x)) # 0.7216325609518421
