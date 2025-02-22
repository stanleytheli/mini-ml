import os
import gzip
import numpy as np
from utils import vectorized_result

def load_mnist(path, kind='train', vectorized_label = True):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28)

    # normalize dataset
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    
    if vectorized_label:
        labels = [vectorized_result(label) for label in labels]
    
    return [(image, label) for image, label in zip(images, labels)]

def load_data_wrapper(filepath="./data"):
    train_labels_path = os.path.join(filepath, 'train-labels-idx1-ubyte.gz')
    train_images_path = os.path.join(filepath, 'train-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(filepath, 't10k-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(filepath, 't10k-images-idx3-ubyte.gz')

    with gzip.open(test_labels_path, 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(test_images_path, 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(test_labels), 28, 28)

    with gzip.open(train_labels_path, 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(train_images_path, 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(train_labels), 28, 28)

    # normalize images
    train_images = (train_images - np.min(train_images)) / (np.max(train_images) - np.min(train_images))
    test_images = (test_images - np.min(test_images)) / (np.max(test_images) - np.min(test_images))

    # separate training into validation and training
    val_images, train_images = train_images[:10000], train_images[10000:]
    val_labels, train_labels = train_labels[:10000], train_labels[10000:]
    
    # vectorize training Y's
    train_labels = [vectorized_result(label) for label in train_labels]

    # now organize the data
    train_data = [(x, y) for x, y in zip(train_images, train_labels)]
    val_data = [(x, y) for x, y in zip(val_images, val_labels)]
    test_data = [(x, y) for x, y in zip(test_images, test_labels)]

    return (train_data, val_data, test_data) 