import numpy as np
from utils import * 

class FullyConnected:

    def __init__(self, input_size, output_size, activation, regularization):
        """Creates a Fully Connected layer."""
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.activation = activation
        self.regularization = regularization

        self.initialize()
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self):
        """Initializes the weights and biases of this layer to be Gaussian random."""
        self.weights = np.random.randn(self.output_size, self.input_size) / np.sqrt(self.input_size)
        self.bias = np.random.randn(self.output_size, 1)

    def feedforward(self, x):
        """Given an input x, returns the activations of this layer."""
        m = x.shape[1]
        self.prev_a = x

        self.z = np.dot(self.weights, x) + np.dot(self.bias, np.ones((1, m)))
        self.a = self.activation.fn(self.z)
        
        return self.a
    
    def update(self, delta):
        """Given the unscaled error deltas of this layer, updates 
        learnable parameters then returns the unscaled error deltas 
        of the previous layer. Input: delta^l, Output: delta^l-1"""
        # Input: unscaled delta^l
        # f'^l(z^l)
        fp = self.activation.derivative(self.z)
        # scaled delta^l
        delta = delta * fp

        # dC/dw^l 
        nabla_w = np.dot(delta, self.prev_a.transpose())
        if self.regularization:
            nabla_w += self.regularization.derivative(self.weights)
        # dC/db^l
        nabla_b = np.sum(delta, axis=1, keepdims=True) #sum over all training examples

        # update learnables
        weights_upd, bias_upd = self.optimizer.fn(nabla_w, nabla_b)
        self.weights += weights_upd
        self.bias += bias_upd

        # return the unscaled delta^l-1
        return self.backpropagate(delta)

    def backpropagate(self, delta):
        """Given scaled delta^l, returns unscaled delta^l-1."""
        return np.dot(self.weights.transpose(), delta)