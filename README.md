# Mini Deep Learning Library

A mini deep learning library built from scratch  using only NumPy/SciPy for calculations. High level API allows for quickly and easily making and training models. 
## Loading Data
Included already are the MNIST and Fashion-MNIST datasets.

```python
mnist_train, mnist_val, mnist_test = mnist_loader.load_data_wrapper("../data/mnist.pkl.gz")
# Because of how the data is structured, the Fashion-MNIST loader 
# takes in the folder path instead of a file path.
fmnist_train, fmnist_val, fmnist_test = fashion_mnist_loader.load_data_wrapper("../data")
```

## Creating a Network
### Initialize
The Network class takes in a list of Layers. Data is processed sequentially, in order, through each Layer. Let's first Flatten the data, then pass it through a 784x30x30x10 network. We'll use a Softmax activation function at the end to normalize our activations.

We'll also add L1 regularization and L2 regularization our layers to combat overfitting and encourage weights to go to 0.
```python
reg1 = L1Regularization(4e-5)
reg2 = L2Regularization(3e-5)

# Small 784x30x30x10 DNN
small_model = modular_network.Network([
    Flatten((28, 28)),
    FullyConnected(28*28, 30, tanh(), reg1),
    FullyConnected(30, 30, tanh(), reg2),
    FullyConnected(30, 10, Softmax(), reg2)
])
```
We can also create Convolutional networks. Let's create a CNN that applies 8 5x5 filters, maxpools, applies 8 3x3s, maxpools, then feeds into a 30x30x10 DNN. We use the correct2Dinput flag to tell the first Convolution Layer that it's going to receive 2d inputs (height, width) instead of 3d (channel, height, width).
```python
# Support for CNNs!
conv_model = modular_network.Network([
    Convolution((28, 28), (5, 5), 8, tanh(), correct2Dinput=True),
    MaxPool((8, 24, 24)),
    Convolution((8, 12, 12), (3, 3), 8, tanh()),
    MaxPool((4, 10, 10)),
    Flatten((4, 5, 5)),
    FullyConnected(4*5*5, 30, tanh(), reg2),
    FullyConnected(30, 30, tanh(), reg2),
    FullyConnected(30, 10, Softmax(), reg2)
])
```
### Learn the Data
Before training, compile the model with a cost function and an optimizer.
```python
small_model.set_cost(BinaryCrossEntropyCost())
# Adam hyperparams: Learning rate 0.005, minibatch size 20, first moment decay factor 0.99, second moment decay factor 0.999
rmsprop = Adam_optimizer(0.005, 20, 0.99, 0.999)
small_model.set_optimizer(rmsprop)
```



Use the Network's train method to learn the data, which takes in the training data, epochs, and  SGD minibatch size. The cost and accuracy on the training and test data can be tracked during training. 
```python
small_model.train(mnist_train, epochs=10, mini_batch_size=20,  
    test_data=mnist_val,
    monitor_test_cost=True,
    monitor_test_acc=False,
    monitor_training_cost=True,
    monitor_training_acc=False)
```
Data augmentation is also supported. Let's create a data augmenter that translates the data up to 2 pixels horizontally and vertically, then adds Normal(0, 0.3) distributed noise.
```python
translate = TranslationAug(-2, 2, -2, 2)
noise = RandnAug(0, 0.3)
combined = CombinedAug([translate, noise])
```
Now pass in the data augmentation to the train function.
```python
small_model.train(mnist_train, epochs=10, mini_batch_size=20, data_augmentation=combined  
```
### Save and Load
We can save the Network to a JSON file.
```python
small_model.save("../save/new_network.json")
```
Let's load a network.
```python
loaded_model = modular_network.Network.load("../save/larger_network.json")
```
We can test its accuracy and cost using the accuracy function, which returns the number of correct classifications on a dataset.
```python
>>> loaded_model.accuracy(mnist_val)
9817
```
mnist_val contains 10,000 images, so getting 9,817 is a validation accuracy of 98.17%. 