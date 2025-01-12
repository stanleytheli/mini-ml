import numpy as np
import random
import time

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class QuadraticCost:
    def fn(a, y):
        """Return the cost associated with an output ``a`` and the 
        desired output ``y``."""
        return 0.5 * np.linalg.norm(a - y)**2
    
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)

class CrossEntropyCost:
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y)

class L1Regularization:
    def cost(w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return np.sum(np.abs(w))
    
    def derivative(w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return np.sign(w)

class L2Regularization:
    def cost(w):
        """Return the cost of a layer's weights as a function of the weight-matrix ``w``."""
        return 0.5 * np.linalg.norm(w)**2

    def derivative(w):
        """Return the derivative of the regularization as a function of the weight ``w``."""
        return w

class Network:
    def __init__(self, sizes, 
                 cost = CrossEntropyCost,
                 regularization = L2Regularization):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost
        self.regularization = regularization
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
        lmbda = 0,
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
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            
            time2 = time.time()

            print(f"Epoch {j+1} training complete, took {time2 - time1} seconds")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_acc:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_test_cost:
                cost = self.total_cost(test_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on test data: {}".format(cost))
            if monitor_test_acc:
                accuracy = self.accuracy(test_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on test data: {} / {}".format(
                    self.accuracy(test_data), n_test))
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
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
        nabla_b, nabla_w = self.backprop(X, Y)

        self.weights = [w - (eta/m)*nw - (eta*lmbda/n)*self.regularization.derivative(w)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/m)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, X, Y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x, summed over all training examples.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # m is the number of training examples in this minibatch
        m = X.shape[1]

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = X
        activations = [X]
        zs = []

        # Feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + np.dot(b, np.ones((1, m)))
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost.delta(zs[-1], activations[-1], Y)
        # Sum the bias gradients over rows (to get sum over training examples)
        nabla_b[-1] = np.dot(delta, np.ones((m, 1)))
        # Similar process with weights
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagation step
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.multiply(np.dot(self.weights[-l+1].transpose(), delta), sp)
            nabla_b[-l] = np.dot(delta, np.ones((m, 1)))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

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

