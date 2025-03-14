import numpy as np

epsilon = 1e-7
gradientClip = 10

class Mode:
    TRAIN = 0
    """The default mode. Network updates learnables upon calling backprop()."""
    TEST = 1
    """Network does not update learnables. Used to cut calcuations and 
     thus speed up evaluations."""

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e