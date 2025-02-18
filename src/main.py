import mnist_loader
import network
import modular_network
from utils import *
from components import * 

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n_train = len(training_data)

reg = L2Regularization(3.125 / n_train)

net = modular_network.Network(
    [
        #Convolution_Independent((28, 28), (5, 5), 1, tanh(), correct2Dinput=True),
        #Flatten((1, 24, 24)),
        Flatten((28, 28)),
        FullyConnected(28*28, 30, tanh(), reg),
        FullyConnected(30, 30, tanh(), reg),
        FullyConnected(30, 10, Softmax(), reg),
    ], 
    cost=BinaryCrossEntropyCost()
)

optim = SGD_momentum_optimizer(0.005, 20, 0.95)
net.set_optimizer(optim)
net.SGD(training_data, 100, 20, test_data, 
        monitor_training_acc=True, 
        monitor_test_acc=True,
        benchmark=False)
