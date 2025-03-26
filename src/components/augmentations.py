import numpy as np
import scipy.signal as sci
from utils import * 

class DataAugmentation:
    def __init__(self):
        """Create a Data Augmentation."""
        pass
    def fn(self, x):
        """Apply this Data Augmentation to the minibatch ``x``."""
        return x

class CombinedAug(DataAugmentation):
    def __init__(self, aug_list):
        """Creates a sequential data augmentation.
        Augmentations processed in the same order as the list,
        e.g. first element is applied first."""
        self.aug_list = aug_list
    
    def fn(self, x):
        for aug in self.aug_list:
            x = aug.fn(x)
        return x

class RandnAug(DataAugmentation):
    def __init__(self, mean = 0, var = 1):
        """Adds i.i.d. normal random noise to each pixel."""
        self.mean = mean
        self.var = var
    
    def fn(self, x):
        return x + self.mean + np.sqrt(self.var) * np.random.randn(*x.shape)

class RandUniformAug(DataAugmentation):
    def __init__(self, min = 0, max = 1):
        """Adds i.i.d. uniform random noise to each pixel."""
        self.min = min
        self.max = max
    
    def fn(self, x):
        return x + np.random.uniform(self.min, self.max, x.shape)
    
class TranslationAug(DataAugmentation):
    def __init__(self, hMin, hMax, wMin, wMax, padding = 0):
        """Translates the image. Translation in the height and width
        axes are uniform random between their min and maxes (inclusive)
        and applied per minibatch. Pads with value ``padding``."""
        self.hMin = hMin
        self.hMax = hMax
        self.wMin = wMin
        self.wMax = wMax
        self.padding = padding
    
    def fn(self, x):
        delta_h = np.random.randint(self.hMin, self.hMax + 1)
        delta_w = np.random.randint(self.wMin, self.wMax + 1)
        
        m, h, w = x.shape

        padding_h = np.ones((m, abs(delta_h), w)) * self.padding
        padding_w = np.ones((m, h, abs(delta_w))) * self.padding

        # delta_w and delta_h must have opposite sign convention if
        # we want them to represent rightward and upward movement respectively
        # This is because tensor width/height axes naturally go rightward/downward
        if delta_h > 0:
            moved_x = x[:, delta_h:h, :]
            x = np.concatenate([moved_x, padding_h], axis=1)
        elif delta_h < 0:
            moved_x = x[:, 0:(h+delta_h), :]
            x = np.concatenate([padding_h, moved_x], axis=1)

        if delta_w > 0:
            moved_x = x[:, :, 0:(w-delta_w)]
            x = np.concatenate([padding_w, moved_x], axis=2)
        elif delta_w < 0:
            moved_x = x[:, :, -delta_w:w]
            x = np.concatenate([moved_x, padding_w], axis=2)

        return x