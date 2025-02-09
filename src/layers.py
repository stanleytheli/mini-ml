import numpy as np
import scipy.signal as sci
from utils import * 

class Layer:
    def __init__(self):
        self.trainable = True
        pass
    def set_optimizer(self, optimizer):
        pass
    def initialize(self):
        pass
    def feedforward(self, x):
        return x
    def backprop(self, delta):
        return delta

class Convolution(Layer):
    def __init__(self, input_shape, filter_shape, filters, activation, 
                 regularization=None, correct2Dinput = False):
        """input_shape = (channels, height, width) or (height, width)
        filter_shape = (height, width)
        filters = number of channels/filters. 
        output shape = (input_channels*filters, height, width)
        Creates a convolutional layer."""
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.filter_shape = filter_shape
        self.num_filters = filters
        self.filters = [np.zeros(filter_shape) for i in range(filters)]
        self.biases = [np.zeros(filter_shape) for i in range(filters)]
        self.regularization = regularization
        self.activation = activation
        self.correct2Dinput = correct2Dinput

        #self.z = [0] * filters
        #self.a = [0] * filters

        self.initialize()
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self):
        for i in range(len(self.filters)):
            self.filters[i] = np.random.randn(*self.filter_shape)
            self.biases[i] = np.random.randn()
    
    def feedforward(self, x):
        """Given a list of m training examples which have shape (channels, height, width),
        calculates their convolutions and returns in the form of a list of m tensors 
        with shape (channels*filters, height', width')."""
        m = len(x)

        if self.correct2Dinput:
            for i in range(m):
                x[i] = np.array([x[i]])

        self.prev_a = x
        self.z = [0] * m
        self.a = [0] * m

        for i in range(m):
            # x[i] is current training example of shape 
            convoluted_channels = [0] * self.input_channels * self.num_filters
            for c in range(self.input_channels):
                for f in range(self.num_filters):
                    # maybe it should really be called a "correlation layer"...
                    convoluted_channels[c * self.num_filters + f] = \
                    sci.correlate(x[i][c], self.filters[f], mode="valid") + self.biases[f] 
                    # to retrieve c: c = index // self.num_filters
                    # to retrieve f: f = index mod self.num_filters
                    # channel c first index is c * self.num_filters
            convoluted = np.stack(convoluted_channels, axis=0)
            self.z[i] = convoluted
            self.a[i] = self.activation.fn(convoluted)

        return self.a

        #for i in range(self.num_filters):
        #    self.z[i] = sci.convolve2d(x, self.filters[i], mode="valid") + self.biases[i]
        #    self.a[i] = self.activation.fn(self.z[i])
    
    def backprop(self, delta):
        """Given a list of m unscaled error deltas, each of shape 
        (channels, height, width), updates this layer's learnables and 
        returns the unscaled error deltas of the previous layer as 
        another list of m deltas each with shape (channels/filters, height, width)"""
        m = len(delta)

        # scale the error deltas
        for i in range(m):
            delta[i] = delta[i] * self.activation.derivative(self.z[i])
        
        nabla_k = np.array([np.zeros(self.filter_shape) for _ in range(self.num_filters)])
        nabla_b = np.zeros((self.num_filters,))

        # calculate gradients
        for f in range(self.num_filters):
            gradientClip = 10

            # dC/dk
            nablas = [
                [sci.correlate(self.prev_a[i][c], delta[i][c * self.num_filters + f], mode="valid")
                 for c in range(self.input_channels)
                ] for i in range(m)
            ]
            nabla_k[f] = np.clip(np.sum(nablas, axis=(0, 1)), -gradientClip, gradientClip)

            # dC/db
            nablas = [
                [np.sum(delta[i][c * self.num_filters + f])
                 for c in range(self.input_channels)
                ] for i in range(m)
            ]
            nabla_b[f] = np.clip(np.sum(nablas, axis=(0, 1)), -gradientClip, gradientClip)

            # PROBLEM with below code: MESSES UP MOMENTUM! 
            # all backprop() functions must call self.optimizer.fn ONCE at most

            # update learnables
            #weights_upd, bias_upd = self.optimizer.fn(nabla_k, nabla_b)

            #self.filters[f] += weights_upd
            #self.biases[f] += bias_upd

        weights_upd, biases_upd = self.optimizer.fn(nabla_k, nabla_b)

        # update learnables
        for f in range(self.num_filters):
            self.filters[f] += weights_upd[f]
            self.biases[f] += biases_upd[f]

        # calculate previous layer's unscaled error deltas
        previous_deltas = [0] * self.input_channels
        for c in range(self.input_channels):
            conved_deltas = [
                [sci.correlate(np.fliplr(np.flipud(self.filters[f])), \
                              delta[i][c*self.num_filters+f], \
                                mode="full")
                for f in range(c * self.num_filters, (c + 1) * self.num_filters)
                ] for i in range(m)
            ]
            previous_deltas[c] = np.sum(conved_deltas, axis=(0, 1))
        return previous_deltas
            
class MaxPool(Layer):
    def __init__(self, input_shape, pool_shape=(2,2)):
        """input_shape = (channels, height, width). 
        pool_shape = (height, width).
        Creates a MaxPool layer. """
        self.input_shape = input_shape
        self.pool_shape = pool_shape

        if len(input_shape) == 2:
            self.mode_2d = True

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