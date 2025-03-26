import numpy as np

class Regularization:
    def __init__(self):
        """Create a regularization."""
        pass
    def cost(self, w):
        """Returns the cost associated with regularization
        on learnable parameters ``w``."""
        return 0
    def derivative(self, w):
        """Returns the derivative of the regularization cost
        at learnable parameters ``w``"""
        return np.zeros(w.shape)

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
