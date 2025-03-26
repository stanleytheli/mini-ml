import numpy as np
import scipy.signal as sci
from utils import * 

class Layer:
    def __init__(self):
        """Create a Layer."""
        self.mode = Mode.TRAIN
        pass
    def set_optimizer(self, optimizer):
        """Set this Layer's optimizer to an Optimizer object
        (NOT and Optimizer factory class)."""
        self.optimizer = optimizer
    def set_mode(self, mode):
        """Set this Layer's mode."""
        self.mode = mode
    def initialize(self):
        """Initialize this Layer's learnable parameters."""
        pass
    def feedforward(self, x):
        """Forward propagate a minibatch ``x``."""
        return x
    def backprop(self, delta):
        """Backward propagate a minibatch of error deltas ``delta``,
        updating learnable parameters on the way if the network is in
        ``Mode.TRAIN``."""
        return delta
    def get_reg_loss(self):
        """Get the cost associated with regularization on this Layer."""
        return 0
    def save_data(self):
        """Get this Layer's learnable parameters in the form of a dictionary."""
        return {}
    def load_data(self, data):
        """Load this Layer's learnable parameters from the dictionary ``data``."""
        pass

class Convolution(Layer):
    def __init__(self, input_shape, filter_shape, filters, activation, 
                 regularization=None, correct2Dinput = False):
        """input_shape = (channels, height, width) or (height, width).
        IMPORTANT: turn on correct2Dinput=True if input_shape is (height, width).
        backprop does NOT work with correct2Dinput (so correct2Dinput only works in first layer).
        filter_shape = (height, width).
        filters = number of channels/filters. 
        output shape = (filters, height, width).
        Creates a convolutional layer."""
        if correct2Dinput:
            input_shape = (1, *input_shape)

        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.filter_shape = filter_shape
        self.num_filters = filters
        self.filters = np.zeros((self.num_filters, self.input_channels, *filter_shape)) # (F, C, h_f, w_f)
        self.biases = np.zeros((self.num_filters,)) # (F,)
        self.regularization = regularization
        self.activation = activation
        self.correct2Dinput = correct2Dinput

        self.initialize()

    def initialize(self):
        self.filters = np.random.randn(*self.filters.shape) / np.sqrt(np.prod(self.filter_shape))
        self.biases = np.random.randn(*self.biases.shape)
    
    def feedforward(self, x):
        """Given an ndarray x of shape (batch, channels, height, width),
        returns convolutions of shape (batch, channels*filters, height', width')
        where height' = height - filter_height + 1 and analogously for width"""
        if self.correct2Dinput:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        
        # x = (m, C, H, W)
        m, C, H, W = x.shape
        F = self.num_filters
        h_f, w_f = self.filter_shape
        Hp, Wp = H - h_f + 1, W - w_f + 1

        self.prev_a = x
        self.z = np.zeros((m, F, Hp, Wp))

        for f in range(F):
            filter = self.filters[f] # (C, h_f, w_f)
            filter = filter.reshape((1, *filter.shape)) # (1, C, h_f, w_f)
            # (M, C, H, W) corr (1, C, h_f, w_f) --> (M, 1, Hp, Wp)
            self.z[:, f] = sci.correlate(x, filter, mode="valid")[:, 0] + self.biases[f] 
        
        self.a = self.activation.fn(self.z)

        return self.a

    def backprop(self, delta):
        C = self.input_channels
        m, F, Hp, Wp = delta.shape
        H, W = self.input_shape[1:]
        h_f, w_f = self.filter_shape

        # scaled error delta
        delta = delta * self.activation.derivative(self.z) # (M, F, Hp, Wp)
        delta_bar = delta.reshape((m, F, 1, Hp, Wp)) # (M, F, 1, Hp, Wp)

        # calculate previous layer deltas
        prev_delta = np.zeros((m, C, H, W))
        for m_i in range(m):
            prev_delta_m = np.zeros((C, H, W))
            for f in range(F):
                filter_R = np.flip(self.filters[f], axis=(1, 2)) # (C, h_f, w_f) flipped across h_f,w_f
                # (C, h_f, w_f) full_corr (1, Hp, Wp) --> (C, H, W)
                prev_delta_m += sci.correlate(filter_R, delta_bar[m_i, f], mode="full")
            prev_delta[m_i] = prev_delta_m

        if self.mode == Mode.TRAIN:
            # dL/dk_f 
            nabla_w = np.zeros((F, C, h_f, w_f))
            for f in range(F):
                # (M, C, H, W) corr (M, 1, Hp, Wp) --> (1, C, h_f, w_f)
                nabla_w[f] = sci.correlate(self.prev_a, delta_bar[:, f], mode="valid")[0]
            if self.regularization:
                nabla_w += self.regularization.derivative(self.filters)

            # dL/db
            nabla_b = np.sum(delta, axis=(0, 2, 3))

            nabla_w = np.clip(nabla_w, -gradientClip, gradientClip)
            nabla_b = np.clip(nabla_b, -gradientClip, gradientClip)

            # update learnables
            weights_upd, biases_upd = self.optimizer.fn([nabla_w, nabla_b])
            self.filters += weights_upd
            self.biases += biases_upd

        return prev_delta
    
    def get_reg_loss(self):
        if self.regularization:
            return self.regularization.cost(self.filters)
        return 0
    
    def save_data(self):
        return {"filters": self.filters.tolist(),
                "biases": self.biases.tolist()}
    
    def load_data(self, data):
        self.filters = np.array(data["filters"])
        self.biases = np.array(data["biases"])

        
class MaxPool(Layer):
    def __init__(self, input_shape, pool_shape=(2,2)):
        """input_shape = (channels, height, width).
        Or just (height, width). pool_shape = (height, width).
        Creates a MaxPool layer. 
        IMPORTANT: If 2D input, and there is a Flatten
        layer, make sure that the Flatten layer has 1 as its channels count. 
        """
        self.correct2Dinput = False
        if len(input_shape) == 2:
            self.correct2Dinput = True
            input_shape = (1, *input_shape)

        self.C, self.H, self.W = input_shape
        self.pool_h, self.pool_w = pool_shape

    def feedforward(self, x):
        if self.correct2Dinput:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # x = (M, C, H, W)
        M = x.shape[0]
        Hp = self.H // self.pool_h
        Wp = self.W // self.pool_w
        x_windowed = x.reshape(M, self.C, Hp, self.pool_h, Wp, self.pool_w)
        maxes = x_windowed.max(axis=(3, 5)) # (M, C, Hp, Wp)

        max_searcher = maxes.repeat(self.pool_h, axis=2).repeat(self.pool_w, axis=3) # (M, C, H, W)
        self.where_max = np.equal(x, max_searcher)

        return maxes
    
    def backprop(self, delta):
        delta_spread = delta.repeat(self.pool_h, axis=2).repeat(self.pool_w, axis=3)
        return delta_spread * self.where_max

class Flatten(Layer):
    def __init__(self, input_shape):
        """input_shape = (channels, height, width).
        Or just (height, width) for 2D flattening.
        Creates a flattener layer that flattens inputs
        of shape (batch, channels, height, width) 
        --> (channels*height*width, m)"""
        self.input_shape = input_shape
        self.flattened_size = np.prod(input_shape)
    
    def feedforward(self, x):
        """Given an ndarray x with shape (batch, input_shape*),
        flattens each training example, then concatenates into a matrix
        with shape (prod(input_shape), m)."""
        self.m = x.shape[0]
        return np.reshape(x, (self.m, self.flattened_size)).T

    def backprop(self, delta):
        """Given the unscaled error deltas, reshapes them to be compatible
        with the layer before the Flatten."""
        return np.reshape(delta.T, (self.m, *self.input_shape))

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

    def initialize(self):
        """Initializes the weights and biases of this layer to be Gaussian random."""
        self.weights = np.random.randn(self.output_size, self.input_size) / np.sqrt(self.input_size)
        self.bias = np.random.randn(self.output_size, 1)

    def feedforward(self, x):
        """Given an input x, returns the activations of this layer."""
        self.prev_a = x

        self.z = np.dot(self.weights, x) + self.bias
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

        # compute prev_delta BEFORE updating weights!
        prev_delta = np.dot(self.weights.transpose(), delta)

        if self.mode == Mode.TRAIN:
            # dC/dw^l 
            nabla_w = np.dot(delta, self.prev_a.transpose())
            if self.regularization:
                nabla_w += self.regularization.derivative(self.weights)
            # dC/db^l
            nabla_b = np.sum(delta, axis=1, keepdims=True) #sum over all training examples

            nabla_w = np.clip(nabla_w, -gradientClip, gradientClip)
            nabla_b = np.clip(nabla_b, -gradientClip, gradientClip)

            # update learnables
            weights_upd, bias_upd = self.optimizer.fn([nabla_w, nabla_b])

            self.weights += weights_upd
            self.bias += bias_upd

        # return the unscaled delta^l-1
        return prev_delta
    
    def get_reg_loss(self):
        if self.regularization:
            return self.regularization.cost(self.weights)
        return 0
    
    def save_data(self):
        return {"weights": self.weights.tolist(),
                "bias": self.bias.tolist()}
    
    def load_data(self, data):
        self.weights = np.array(data["weights"])
        self.bias = np.array(data["bias"])


class FullyConnectedPostbias(FullyConnected):
    def __init__(self, input_size, output_size, activation, regularization):
        """Much like a FullyConnected layer, but with another set of biases 
        applied *after* the activation function."""
        self.postbias = np.zeros((output_size, 1))
        super().__init__(input_size, output_size, activation, regularization)

    def initialize(self):
        super().initialize()
        self.postbias = np.random.randn(self.output_size, 1)
    
    def feedforward(self, x):
        return super().feedforward(x) + self.postbias

    def backprop(self, delta):
        # conveniently the unscaled error deltas are dC/da^l

        # dC/dB^l 
        nabla_beta = np.sum(delta, axis=1, keepdims=True) #sum over all training examples
        
        fp = self.activation.derivative(self.z)
        # scaled delta^l = dC/dz^l
        delta = delta * fp

        # compute prev_delta BEFORE updating weights!
        prev_delta = np.dot(self.weights.transpose(), delta)

        # dC/dw^l 
        nabla_w = np.dot(delta, self.prev_a.transpose())
        if self.regularization:
            nabla_w += self.regularization.derivative(self.weights)
        # dC/db^l
        nabla_b = np.sum(delta, axis=1, keepdims=True) #sum over all training examples

        nabla_w = np.clip(nabla_w, -gradientClip, gradientClip)
        nabla_b = np.clip(nabla_b, -gradientClip, gradientClip)
        nabla_beta = np.clip(nabla_beta, -gradientClip, gradientClip)

        # update learnables
        weights_upd, bias_upd, postbias_upd = self.optimizer.fn([nabla_w, nabla_b, nabla_beta])

        self.weights += weights_upd
        self.bias += bias_upd
        self.postbias += postbias_upd

        # return the unscaled delta^l-1
        return prev_delta

    
    def save_data(self):
        return {"weights": self.weights.tolist(),
                "bias": self.bias.tolist(),
                "postbias": self.postbias.tolist()}
    
    def load_data(self, data):
        self.weights = np.array(data["weights"])
        self.bias = np.array(data["bias"])
        self.postbias = np.array(data["postbias"])

