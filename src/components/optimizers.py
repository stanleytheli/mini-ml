import numpy as np
from utils import *

# Factory class
class SGD_optimizer:
    def __init__(self, eta, m):
        """eta: learning rate; m: minibatch size"""
        self.eta = eta
        self.m = m
    
    def get_optimizer(self):
        return self._optimizer(self.eta, self.m)

    class _optimizer:
        def __init__(self, eta, m):
            self.eta = eta
            self.m = m
        
        def fn(self, learnables):
            """Given a list learnables of ndarrays representing the gradient
            wrt. the learnable parameters, returns the updates in the same form."""
            return [-(self.eta/self.m)*nabla for nabla in learnables]

class SGD_momentum_optimizer:
    def __init__(self, eta, m, beta):
        """eta: learning rate; m: minibatch size; beta: momentum parameter"""
        self.eta = eta
        self.m = m
        self.beta = beta

    def get_optimizer(self):
        return self._optimizer(self.eta, self.m, self.beta)

    class _optimizer:
        def __init__(self, eta, m, beta):
            self.eta = eta
            self.m = m
            self.beta = beta

            self.v = None

        def fn(self, learnables):
            """Given a list learnables of ndarrays representing the gradient
            wrt. the learnable parameters, returns the updates in the same form."""
            # Forgot the minus sign and spent a few minutes wondering what 
            # was wrong with my network. Turns out it was doing gradient ascent!
            if self.v is not None:
                self.v = [ - (self.beta*v_i + (self.eta/self.m)*nabla_i) 
                        for v_i, nabla_i in zip(self.v, learnables)]
            else:
                self.v = [-(self.eta/self.m)*nabla for nabla in learnables]
            return self.v

class RMSProp_optimizer:
    def __init__(self, eta, m, beta):
        self.eta = eta
        self.m = m
        self.beta = beta

    def get_optimizer(self):
        return self._optimizer(self.eta, self.m, self.beta)
    
    class _optimizer:
        def __init__(self, eta, m, beta):
            self.eta = eta
            self.m = m
            self.beta = beta

            self.v = None

        def fn(self, learnables):
            # Exponentially decaying moving average
            if self.v is not None:
                self.v = [self.beta * v_i + (1 - self.beta) * np.square(nabla_i)
                          for v_i, nabla_i in zip(self.v, learnables)]
            else:
                self.v = [(1 - self.beta) * np.square(nabla) for nabla in learnables]

            updates = [ - (self.eta/self.m)*(nabla_i/(np.sqrt(v_i) + epsilon))
                       for v_i, nabla_i in zip(self.v, learnables)]
            return updates

class Adam_optimizer:
    def __init__(self, eta, m, beta_1, beta_2):
        self.eta = eta
        self.m = m
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def get_optimizer(self):
        return self._optimizer(self.eta, self.m, self.beta_1, self.beta_2)
    
    class _optimizer():
        def __init__(self, eta, m, beta_1, beta_2):
            self.eta = eta
            self.m = m
            self.beta_1 = beta_1
            self.beta_2 = beta_2

            self.a = None
            self.v = None
            self.t = 0

        def fn(self, learnables):
            self.t += 1

            if self.a is not None:
                self.a = [self.beta_1 * a_i + (1 - self.beta_1) * nabla_i 
                          for a_i, nabla_i in zip(self.a, learnables)]
            else:
                self.a = [(1 - self.beta_1) * nabla for nabla in learnables]
            
            if self.v is not None:
                self.v = [self.beta_2 * v_i + (1 - self.beta_2) * np.square(nabla_i)
                          for v_i, nabla_i in zip(self.v, learnables)]
            else:
                self.v = [(1 - self.beta_2) * np.square(nabla) for nabla in learnables]

            bias_correction = np.sqrt(1 - np.power(self.beta_2, self.t))/(1 - np.power(self.beta_1, self.t))
            return [- (self.eta/self.m) * bias_correction * (a_i/(np.sqrt(v_i) + epsilon))
                    for a_i, v_i in zip(self.a, self.v)]