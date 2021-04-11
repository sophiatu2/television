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

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)
        
        self.architecture = [
               # Match con2d,maxpool2x -> next con2d args
               #Start smaller?
            # Block 1
            Conv2D(32, 3, 1, padding="same", activation="relu"),
            Conv2D(32, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2),
            # Block 2
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2),
            # Block 3
            Conv2D(128, 3, 1, padding="same", activation="relu"),
            Conv2D(128, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2),
            # Last Layers
            Conv2D(256, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2),
            Dropout(0.05),
            Flatten(),
            #Dense takes in num classes
            Dense(128,activation="relu"),
            Dense(hp.num_classes, activation = "softmax")
        ]
             

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        return tf.keras.losses.sparse_categorical_crossentropy(y_true = labels, y_pred = predictions, from_logits=False)

