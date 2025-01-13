import mnist_loader
import network
from utils import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 100, 100, 10], 
                      [clippedReLU, clippedReLU, Softmax],
                      cost = CrossEntropyCost,
                      regularization = L2Regularization)
net.SGD(training_data, 60, 20, 0.01, 
        mu = 0.9,
        lmbda = 5.0,
        test_data = test_data, 
        monitor_test_acc = True)
net.save("./save/network-1.json")