import numpy as np
from utils import * 

class Layer:
    def __init__(self):
        pass
    def set_optimizer(self, optimizer):
        pass
    def initialize(self):
        pass
    def feedforward(self, x):
        return x
    def backprop(self, delta):
        return delta

class Flatten(Layer):
    def __init__(self, input_shape):
        """input_shape = (width, height, channels).
        Creates a flattener layer that flattens lists of m 
        inputs (width, height, channels) --> a single matrix
        with shape (width*height*channels, m)"""

        # input_shape and output_shape are shapes of the individual vectors,
        # NOT the shapes of the matrices that are the actual outputs

        w, h, c = input_shape
        self.input_shape = input_shape
        self.output_shape = (w * h * c, 1)
    
    def feedforward(self, x):
        """Given a list x containing m training examples which have shape input_shape,
        flattens each training example, then concatenates into a single matrix."""
        m = len(x)
        
        a = [0] * m
        for i in range(m):
            a[i] = np.reshape(x[i], self.output_shape)
        
        a = np.concatenate(a, axis=1)
        return a

    def backprop(self, delta):
        """Given the unscaled error deltas, reshapes them to be compatible
        with the layer before the Flatten."""
        m = delta.shape[1]

        reshaped = [0] * m
        delta_list = np.split(delta, range(m), axis=1)[1:]
        for i in range(m):
            reshaped[i] = np.reshape(delta_list[i], self.input_shape)

        return reshaped

class FullyConnected(Layer):
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
    
    def backprop(self, delta):
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

        gradientClip = 10
        nabla_w = np.clip(nabla_w, -gradientClip, gradientClip)
        nabla_b = np.clip(nabla_b, -gradientClip, gradientClip)

        # update learnables
        weights_upd, bias_upd = self.optimizer.fn(nabla_w, nabla_b)

        #if self.regularization:
        #    weights_upd -= 0.001 * self.regularization.derivative(self.weights)

        self.weights += weights_upd
        self.bias += bias_upd

        # return the unscaled delta^l-1
        return np.dot(self.weights.transpose(), delta)
