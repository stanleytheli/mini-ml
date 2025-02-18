import numpy as np
from utils import epsilon

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
