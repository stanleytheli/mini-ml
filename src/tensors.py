import tensorflow as tf

import keras
from keras import datasets, layers, activations, regularizers, optimizers, losses

fashion_mnist = datasets.mnist
(unnorm_train_images, train_labels), (unnorm_test_images, test_labels) = fashion_mnist.load_data()
train_images = unnorm_train_images / 255
test_images = unnorm_test_images / 255

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation=activations.celu, input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation=activations.celu),
    keras.layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(100, activation=activations.celu, 
                 kernel_regularizer=regularizers.l2(0.00001)),
    layers.Dense(100, activation=activations.celu,
                 kernel_regularizer=regularizers.l2(0.00001)),
    layers.Dense(10, activation=activations.softmax),
])

model.compile(optimizer = optimizers.Adam(),
              loss = losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"],
              )

model.fit(train_images, train_labels, epochs=30, batch_size=32)

model.evaluate(test_images, test_labels)