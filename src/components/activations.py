import numpy as np

class noActivation:
    def fn(self, z):
        return z
    
    def derivative(self, z):
        return np.ones(z.shape)

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
        return np.clip(z, 0)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0)

class clippedReLU:
    def __init__(self, clip = 5):
        self.clip = clip
    
    def fn(self, z):
        return np.clip(z, 0, self.clip)
    
    def derivative(self, z):
        return np.greater_equal(z, 0.0) * np.less_equal(z, self.clip)

class Softmax:
    def fn(self, z):
        z = z - np.max(z) # For numerical stability
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def derivative(self, z):
        return self.fn(z) * (1 - self.fn(z))
