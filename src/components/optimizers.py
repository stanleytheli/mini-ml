import numpy as np

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

