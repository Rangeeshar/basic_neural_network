import numpy as np

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return self.sigmoid(total)
    
  def sigmoid(self, x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


if __name__=="__main__":
    try:
        val1, val2 = map(int, input("Enter 2 Inputs for neuron: ").split())
        w1, w2 = map(int, input("Enter 2 Weights for neuron: ").split())
        bias = int(input("Enter Bias : "))
        weights = np.array([w1, w2]) 
        inputs = np.array([val1, val2])
    except Exception as e:
        print("Error: {}".format(e))
    n = Neuron(weights, bias)
    print("Output from our simple neuron : {}".format(n.feedforward(inputs))) 
