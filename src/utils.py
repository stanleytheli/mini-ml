import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class QuadraticCost:
    def fn(a, y):
        """Return the cost associated with an output ``a`` and the 
        desired output ``y``."""
        return 0.5 * np.linalg.norm(a - y)**2
    
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)

class CrossEntropyCost:
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y)

class L1Regularization:
    def cost(w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return np.sum(np.abs(w))
    
    def derivative(w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return np.sign(w)

class L2Regularization:
    def cost(w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return 0.5 * np.linalg.norm(w)**2

    def derivative(w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return w
    
class L1PlusL2Regularization:
    # L1PlusL2 = L1 + alpha * L2. 
    alpha = 1

    def cost(w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return L1Regularization.cost(w) + L1PlusL2Regularization.alpha * L2Regularization.cost(w)
    
    def derivative(w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return L1Regularization.derivative(w) + L1PlusL2Regularization.alpha * L2Regularization.derivative(w)
