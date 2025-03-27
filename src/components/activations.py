import numpy as np

class ActivationFunction:
    def __init__(self):
        self.init_params = []
    def fn(self, z):
        """Returns the value of this Activation Function 
        evaluated at logits ``z``."""
        return z
    def derivative(self, z):
        """Returns the derivative of this Activation Function
        at logits ``z``."""
        return np.ones(z.shape)
    def save_construction(self):
        return {"name": self.__class__.__name__, "params": self.init_params}

class noActivation(ActivationFunction):
    pass

class Sigmoid(ActivationFunction):
    def fn(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def derivative(self, z):
        return self.fn(z)*(1 - self.fn(z))

class tanh(ActivationFunction):
    def fn(self, z):
        return np.tanh(z)
    
    def derivative(self, z):
        return 1 - np.tanh(z) * np.tanh(z)

class ReLU(ActivationFunction):
    def fn(self, z):
        return np.clip(z, 0)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0)

class clippedReLU(ActivationFunction):
    def __init__(self, clip = 5):
        self.clip = clip
        self.init_params = [clip]
    
    def fn(self, z):
        return np.clip(z, 0, self.clip)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0) * np.less_equal(z, self.clip)

class Softmax(ActivationFunction):
    def fn(self, z):
        z = z - np.max(z) # For numerical stability
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def derivative(self, z):
        return self.fn(z) * (1 - self.fn(z))
