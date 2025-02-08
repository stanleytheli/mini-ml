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

class MaxPool(Layer):
    def __init__(self, input_shape, pool_shape=(2,2)):
        """input_shape = (channels, height, width). 
        pool_shape = (height, width).
        Creates a MaxPool layer. """
        self.input_shape = input_shape
        self.pool_shape = pool_shape

    def feedforward(self, x):
        """Given a list of m training examples which have shape (channels, height, width),
        calculates their MaxPool and returns in the form of a list of m tensors 
        with shape (channels/stride, height/stride, width)."""
        # x is a list of m images each of shape(c, h, w)
        m = len(x)

        C, H, W = self.input_shape
        pool_h, pool_w = self.pool_shape

        h_steps = H // pool_h
        w_steps = W // pool_w

        # Is there a better way to keep track of max values for backprop while
        # avoiding python for loops?
        pooled = [0] * m
        self.max_indices = [0] * m

        for i in range(m):
            pooled_image = np.zeros((C, h_steps, w_steps))
            max_idxs = np.zeros((C, h_steps, w_steps, 2), dtype=int)

            for h in range(h_steps):
                for w in range(w_steps):
                    h_start, h_end = h * pool_h, (h + 1) * pool_h
                    w_start, w_end = w * pool_w, (w + 1) * pool_w
                    window = x[i][:, h_start:h_end, w_start:w_end]

                    max_vals = window.max(axis=(1,2))
                    pooled_image[:, h, w] = max_vals

                    for c in range(C):
                        max_idx = np.argwhere(window[c] == max_vals[c])[0]
                        max_idxs[c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]

            pooled[i] = pooled_image
            self.max_indices[i] = max_idxs
        
        return pooled
        """
        pooled = [0] * m
        for i in range(m):
            pool_batch = [0] * self.channels
            for c in range(self.channels):
                image = x[i][:, :, c]
                windows = image.reshape(self.rows//self.pool_shape[0], self.pool_shape[0],
                            self.cols//self.pool_shape[1], self.pool_shape[1])
                pool_batch[c] = windows.max((1, 3))
            pooled[i] = np.stack(pool_batch, axis=2)
        return pooled
        """
    
    def backprop(self, delta):
        """Given the unscaled error deltas [in the form of a list of m error 
        deltas which each have shape (channels, height/stride, width/stride)] reshapes 
        them to be compatible with the layer before the MaxPool."""
        m = len(delta)

        C, H, W = self.input_shape
        pool_h, pool_w = self.pool_shape

        h_steps = H // pool_h
        w_steps = W // pool_w

        deltas_list = [0] * m
        for i in range(m):
            reshaped_deltas = np.zeros((C, H, W))

            for h in range(h_steps):
                for w in range(w_steps):
                    for c in range(C):
                        h_idx, w_idx = self.max_indices[i][c, h, w]
                        reshaped_deltas[c, h_idx, w_idx] = delta[i][c, h, w]

            deltas_list[i] = reshaped_deltas
        
        return deltas_list


class Flatten(Layer):
    def __init__(self, input_shape):
        """input_shape = (channels, height, width).
        Or just (height, width) for 2D flattening.
        Creates a flattener layer that flattens lists of m 
        inputs (channels, height, width) --> a single matrix
        with shape (channels*height*width, m)"""

        # input_shape and output_shape are shapes of the individual vectors,
        # NOT the shapes of the matrices that are the actual outputs

        self.input_shape = input_shape
        flattened_size = np.prod(input_shape)
        self.output_shape = (flattened_size, 1)
    
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
