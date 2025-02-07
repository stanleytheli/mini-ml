import numpy as np
import random
import time
import json
import sys
from utils import *
from layers import *

class Network:

    def __init__(self, layers, cost):
        self.num_layers = len(layers)
        self.layers = layers
        self.cost = cost
    
    def set_optimizer(self, optimizer):
        for layer in self.layers:
            # the optimizer object passed into the modular network is more like
            # an optimizer factory, which is why get_optimizer() exists. 
            # this choice was made to allow compartmentalization of layers.
            layer.set_optimizer(optimizer.get_optimizer())
    
    def feedforward(self, a):
        for layer in self.layers:
            a = layer.feedforward(a)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, 
        test_data=None,
        monitor_test_cost=False,
        monitor_test_acc=False,
        monitor_training_cost=False,
        monitor_training_acc=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            time1 = time.time()

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, len(training_data))
            
            time2 = time.time()

            print(f"Epoch {j+1} training complete, took {time2 - time1} seconds")
            if monitor_training_cost:
                cost = self.total_cost(training_data)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_acc:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_test_cost:
                cost = self.total_cost(test_data, convert=True)
                evaluation_cost.append(cost)
                print("Cost on test data: {}".format(cost))
            if monitor_test_acc:
                accuracy = self.accuracy(test_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on test data: {} / {}".format(
                    self.accuracy(test_data), n_test))
    
    def update_mini_batch(self, mini_batch, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        # m is the number of training examples in each this minibatch
        m = len(mini_batch)

        # Combine x and y vectors of mini batch into a single matrix
        X = np.concatenate([pair[0] for pair in mini_batch], axis=1)
        Y = np.concatenate([pair[1] for pair in mini_batch], axis=1)

        # Backpropagation
        self.backprop(X, Y)

    def backprop(self, X, Y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x, summed over all training examples.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # m is the number of training examples in this minibatch
        m = X.shape[1]

        # a^L
        a_L = self.feedforward(X)
        # unscaled delta^L
        delta = self.cost.derivative(a_L, Y)

        for layer in reversed(self.layers):
            delta = layer.update(delta)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: 
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(self.regularization.cost(w) for w in self.weights)
        return cost

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    ### Saving a Network
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "activations": [str(func.__name__) for func in self.activations],
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__),
                "regularization": str(self.regularization.__name__)}
        
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    #### Loading a Network
    def load(filename):
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network.

        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        regularization = getattr(sys.modules[__name__], data["regularization"])
        activations = [getattr(sys.modules[__name__], func_name) for func_name in data["activations"]]
        net = Network(data["sizes"], 
                      activations,
                      cost=cost, 
                      regularization=regularization)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
