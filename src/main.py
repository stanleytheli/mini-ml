import mnist_loader
import network
import modular_network
from utils import *
from layers import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

reg = L2Regularization(3.125 / 50000)

net = modular_network.Network(
    [
        Convolution((1, 28, 28), (3, 3), 4, noActivation(), correct2Dinput=True),
        #MaxPool((4, 26, 26), (2, 2)),
        #Flatten((4, 13, 13)),
        #FullyConnected(4*13*13, 30, tanh(), reg),
        Flatten((4, 26, 26)),
        FullyConnected(4*26*26, 30, tanh(), reg),
        FullyConnected(30, 30, tanh(), reg),
        FullyConnected(30, 10, Softmax(), reg),
    ], 
    cost=BinaryCrossEntropyCost()
)

optim = SGD_momentum_optimizer(0.005, 20, 0.95)
#optim = SGD_optimizer(0.01, 20)
net.set_optimizer(optim)
net.SGD(training_data, 100, 20, test_data, monitor_training_acc=False, monitor_test_acc=True)

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