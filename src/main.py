import mnist_loader
import network
import modular_network
from utils import *
from layers import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n_train = len(training_data)

reg = L2Regularization(3.125 / n_train)

net = modular_network.Network(
    [
        Convolution_List((1, 28, 28), (3, 3), 8, noActivation(), correct2Dinput=True),
        #MaxPool((8, 26, 26), (2, 2)), --> (8, 13, 13)
        Flatten_List((8, 26, 26)),
        FullyConnected(8*26*26, 100, tanh(), reg),
        FullyConnected(100, 100, tanh(), reg),
        FullyConnected(100, 10, Softmax(), reg),
    ], 
    cost=BinaryCrossEntropyCost()
)

optim = SGD_momentum_optimizer(0.005, 20, 0.95)
net.set_optimizer(optim)
net.SGD(training_data, 100, 20, test_data, 
        monitor_training_acc=False, 
        monitor_test_acc=True,
        benchmark=True)
