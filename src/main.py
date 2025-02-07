import mnist_loader
import network
import modular_network
from utils import *
from layers import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

reg = L2Regularization(0.0000625)

net = modular_network.Network(
    [
        Flatten((28, 28, 1)),
        FullyConnected(28*28, 100, tanh(), reg),
        FullyConnected(100, 100, tanh(), reg),
        FullyConnected(100, 10, Softmax(), reg),
    ], 
    cost=BinaryCrossEntropyCost()
)

optim = SGD_momentum_optimizer(0.005, 20, 0.95)
#optim = SGD_optimizer(0.01, 20)
net.set_optimizer(optim)
net.SGD(training_data, 100, 20, test_data, monitor_training_acc=True, monitor_test_acc=True)

"""
net = network.Network([784, 100, 100, 10], 
                      [clippedReLU(), clippedReLU(), Softmax()],
                      cost = BinaryCrossEntropyCost(),
                      regularization = L2Regularization())
net.SGD(training_data, 60, 20, 0.01, 
        mu = 0.9,
        lmbda = 5.0,
        test_data = test_data, 
        monitor_test_acc = True)
#net.save("./save/network-1.json")"""