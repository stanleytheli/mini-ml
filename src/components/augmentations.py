import numpy as np
import scipy.signal as sci
from utils import * 

class DataAugmentation:
    def __init__(self):
        pass
    def fn(self, x):
        return x

class CombinedAug(DataAugmentation):
    def __init__(self, aug_list):
        self.aug_list = aug_list
    
    def fn(self, x):
        for aug in self.aug_list:
            x = aug.fn(x)
        return x

class RandnAug(DataAugmentation):
    def __init__(self, mean = 0, var = 1):
        self.mean = mean
        self.var = var
    
    def fn(self, x):
        return x + self.mean + np.sqrt(self.var) * np.random.randn(*x.shape)

class RandUniformAug(DataAugmentation):
    def __init__(self, min = 0, max = 1):
        self.min = min
        self.max = max
    
    def fn(self, x):
        return x + np.random.uniform(self.min, self.max, x.shape)