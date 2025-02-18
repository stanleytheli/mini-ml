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

class Convolution_Independent(Layer):
    def __init__(self, input_shape, filter_shape, filters, activation, 
                 regularization=None, correct2Dinput = False):
        """input_shape = (channels, height, width) or (height, width).
        IMPORTANT: turn on correct2Dinput=True if input_shape is (height, width).
        backprop does NOT work with correct2Dinput (so correct2Dinput only works in first layer).
        filter_shape = (height, width).
        filters = number of channels/filters. 
        output shape = (input_channels*filters, height, width).
        Creates a convolutional layer."""
        if correct2Dinput:
            input_shape = (1, *input_shape)

        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.filter_shape = filter_shape
        self.num_filters = filters
        self.filters = np.zeros((self.num_filters, *filter_shape))
        self.biases = np.zeros((self.num_filters,))
        self.regularization = regularization
        self.activation = activation
        self.correct2Dinput = correct2Dinput

        self.initialize()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self):
        self.filters = np.random.randn(*self.filters.shape) / np.sqrt(np.prod(self.filter_shape))
        self.biases = np.random.randn(*self.biases.shape)
    
    def feedforward(self, x):
        """Given an ndarray x of shape (batch, channels, height, width),
        returns convolutions of shape (batch, channels*filters, height', width')
        where height' = height - filter_height + 1 and analogously for width"""
        if self.correct2Dinput:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        m, C, H, W = x.shape
        F = self.num_filters
        h_f, w_f = self.filter_shape
        Hp, Wp = H - h_f + 1, W - w_f + 1

        # reshaped filters for batching purposes
        filters_bar = self.filters.reshape(F, 1, 1, h_f, w_f) 

        self.prev_a = x # (m, C, H, W)
        self.z = np.zeros((m, C * F, Hp, Wp)) # (m, CF, H, W)

        for f in range(F):
            self.z[:, f::C] = sci.correlate(x, filters_bar[f], mode="valid") + self.biases[f]
            # indexing system: input channel c and filter f --> i = c*(n_f) + f

        self.a = self.activation.fn(self.z)

        return self.a

    def backprop(self, delta):
        """Given an ndarray of error deltas of shape 
        (batch, channels*filters, height', width'), updates
        learnables and then returns the previous layer's
        unscaled error deltas in shape (batch, channels, height, width)"""
        C = self.input_channels
        F = self.num_filters
        m, _, Hp, Wp = delta.shape
        H, W = self.input_shape[1:]
        h_f, w_f = self.filter_shape

        # scaled deltas
        delta = delta * self.activation.derivative(self.z)
        # reshaped variables for batching purposes
        delta_bar = delta.reshape((m, C, F, Hp, Wp))
        filters_bar = self.filters.reshape(F, 1, 1, h_f, w_f)
        # filters, reflected across image axes
        filters_bar_R = np.flip(filters_bar, axis=(3, 4)) # (F, 1, 1, h_f, w_f)

        # dC/dw
        nabla_w = np.zeros(self.filters.shape) # (F, h_f, w_f)
        for f in range(F):
            nabla_w[f] = sci.correlate(self.prev_a, delta_bar[:, :, f], mode="valid")[0, 0]
            # (m, C, H, W) corr (m, C, H', W') --> (1, 1, h_f, w_f)
        # dC/db
        nabla_b = np.sum(delta_bar, axis=(0, 1, 3, 4)) # (F,)

        # update learnables
        weights_upd, biases_upd = self.optimizer.fn([nabla_w, nabla_b])
        self.filters += weights_upd
        self.biases += biases_upd

        # unscaled error deltas of previous layer
        previous_deltas = np.zeros((m, C, F, H, W))
        for f in range(F):
            previous_deltas[:, :, f] = sci.correlate(filters_bar_R[f], delta_bar[:, :, f], mode="full")
            # (1, 1, h_f, w_f) full_corr (m, C, H', W') --> (m, C, H, W)
        previous_deltas = np.sum(previous_deltas, axis=2) # (m, C, H, W)
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
        weights_upd, bias_upd = self.optimizer.fn([nabla_w, nabla_b])

        #if self.regularization:
        #    weights_upd -= 0.001 * self.regularization.derivative(self.weights)

        self.weights += weights_upd
        self.bias += bias_upd

        # return the unscaled delta^l-1
        return np.dot(self.weights.transpose(), delta)

class FullyConnectedPostbias(FullyConnected):
    def __init__(self, input_size, output_size, activation, regularization):
        """Much like a FullyConnected layer, but with another set of biases 
        applied *after* the activation function."""
        self.postbias = np.zeros((output_size, 1))
        super().__init__(input_size, output_size, activation, regularization)

    def initialize(self):
        super().initialize()
        self.postbias = np.random.randn(self.output_size, 1)
    
    def feedforward(self, a):
        m = a.shape[1]
        return super().feedforward(a) + np.dot(self.postbias, np.ones((1, m)))

    def backprop(self, delta):
        # conveniently the unscaled error deltas are dC/da^l

        # dC/dB^l 
        nabla_beta = np.sum(delta, axis=1, keepdims=True) #sum over all training examples
        
        fp = self.activation.derivative(self.z)
        # scaled delta^l = dC/dz^l
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
        nabla_beta = np.clip(nabla_beta, -gradientClip, gradientClip)

        # update learnables
        weights_upd, bias_upd, postbias_upd = self.optimizer.fn([nabla_w, nabla_b, nabla_beta])

        self.weights += weights_upd
        self.bias += bias_upd
        self.postbias += postbias_upd

        # return the unscaled delta^l-1
        return np.dot(self.weights.transpose(), delta)

