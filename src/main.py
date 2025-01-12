import mnist_loader
import network
from utils import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10], 
                      regularization=L1Regularization)
net.SGD(training_data, 30, 10, 0.5, 
        lmbda = 5.0,
        test_data = test_data, 
        monitor_test_acc = True)
net.save("./save/network-1.json")