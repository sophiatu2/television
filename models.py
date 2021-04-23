# From project 4

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # Select an optimizer for your network

        self.optimizer = RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)
        
        # Architecture for the CNN
        self.architecture = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block2_pool"),
            Dropout(rate=0.25),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block3_pool"),
            Dropout(rate=0.25),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            MaxPool2D(2, name="block4_pool"),
            Dropout(rate=0.25),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(6, activation="softmax")
        ]
             

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # loss function for your network
        return tf.keras.losses.sparse_categorical_crossentropy(y_true = labels, y_pred = predictions, from_logits=False)