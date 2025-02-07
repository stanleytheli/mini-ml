import numpy as np

epsilon = 1e-7
gradientClip = 1000

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Factory class
class SGD_optimizer:
    def __init__(self, eta, m):
        """eta: learning rate; m: minibatch size"""
        self.eta = eta
        self.m = m
    
    def get_optimizer(self):
        return _SGD(self.eta, self.m)

class _SGD:
    def __init__(self, eta, m):
        self.eta = eta
        self.m = m
    
    def fn(self, nabla_w, nabla_b):
        return - (self.eta/self.m) * nabla_w, - (self.eta/self.m) * nabla_b

class SGD_momentum_optimizer:
    def __init__(self, eta, m, beta):
        """eta: learning rate; m: minibatch size; beta: momentum parameter"""
        self.eta = eta
        self.m = m
        self.beta = beta

    def get_optimizer(self):
        return _SGD_momentum(self.eta, self.m, self.beta)

class _SGD_momentum:
    def __init__(self, eta, m, beta):
        self.eta = eta
        self.m = m
        self.beta = beta

        self.w_v = 0
        self.b_v = 0

    def fn(self, nabla_w, nabla_b):
        # Forgot the minus sign and spent a few minutes wondering what 
        # was wrong with my network. Turns out it was doing gradient ascent!
        self.w_v = - (self.beta * self.w_v + (self.eta/self.m) * nabla_w)
        self.b_v = - (self.beta * self.b_v + (self.eta/self.m) * nabla_b)
        return self.w_v, self.b_v

class Sigmoid:
    def fn(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def derivative(self, z):
        return self.fn(z)*(1 - self.fn(z))

class tanh:
    def fn(self, z):
        return np.tanh(z)
    
    def derivative(self, z):
        return 1 - np.tanh(z) * np.tanh(z)

class ReLU:
    def fn(self, z):
        return z * np.greater_equal(z, 0.0)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0)

class clippedReLU:
    def __init__(self, clip = 20):
        self.clip = clip
    
    def fn(self, z):
        return z * np.greater_equal(z, 0.0) * np.less_equal(z, self.clip) \
                + self.clip * np.greater(z, self.clip)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0) * np.less_equal(z, self.clip)

class Softmax:
    def fn(self, z):
        z = z - np.max(z) # For numerical stability
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def derivative(self, z):
        return self.fn(z) * (1 - self.fn(z))

class QuadraticCost:
    def fn(self, a, y):
        """Return the cost associated with an output ``a`` and the 
        desired output ``y``."""
        return 0.5 * np.linalg.norm(a - y)**2
    
    def derivative(self, a, y):
        """Return the derivative with respect to the activation."""
        return (a - y)

class BinaryCrossEntropyCost:
    def fn(self, a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    def derivative(self, a, y):
        """Return the derivative with respect to the activation."""
        a = np.clip(a, epsilon, 1 - epsilon)
        return (a - y) / (a * (1 - a))

class L1Regularization:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def cost(self, w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return self.lmbda * np.sum(np.abs(w))
    
    def derivative(self, w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return self.lmbda * np.sign(w)

class L2Regularization:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def cost(self, w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return 0.5 * self.lmbda * np.linalg.norm(w)**2

    def derivative(self, w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return self.lmbda * w
    
class L1PlusL2Regularization:
    # L1PlusL2 = alpha * L1 + beta * L2. 
    def __init__(self, lmbda, alpha = 1, beta = 1):
        self.lmbda = lmbda
        self.alpha = alpha
        self.beta = beta

        self.l1 = L1Regularization(lmbda)
        self.l2 = L2Regularization(lmbda)

    def cost(self, w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return self.alpha * self.l1.cost(w) + self.beta * self.l2.cost(w)
    
    def derivative(self, w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return self.alpha * self.l1.derivative(w) + self.beta * self.l2.derivative(w)
