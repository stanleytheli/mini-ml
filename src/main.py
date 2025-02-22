import mnist_loader
import modular_network
from utils import *
from components import * 

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n_train = len(training_data)

reg = L2Regularization(3.125 / n_train)

net = modular_network.Network(
    [
        Convolution((28, 28), (3, 3), 4, tanh(), correct2Dinput = True), # --> (4, 26, 26)
        MaxPool((4, 26, 26)), # --> (4, 13, 13)
        Convolution((4, 13, 13), (2, 2), 4, tanh()), # --> (4, 12, 12)
        MaxPool((4, 12, 12)), # --> (4, 6, 6)
        Flatten((4, 6, 6)),
        FullyConnected(4*6*6, 100, tanh(), reg),
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
        )
