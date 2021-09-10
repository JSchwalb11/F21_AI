import tensorflow as tf


# Neural Network Model
#  Input Layer: 512^2 neurons (512 x 512)
#  Hidden Layer1: 128 neurons
#  Output Layer: 2 neurons (Yes or No) or (1 or 0)
def build_fc_model():
    fc_model = tf.keras.Sequential([
        # First define a Flatten layer
        tf.keras.layers.Flatten(),

        # '''TODO: Define the activation function for the first fully connected (Dense) layer.'''
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # '''TODO: Define the second Dense layer to output the classification probabilities'''
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    ])
    return fc_model


def build_cnn_model():
    cnn_model = tf.keras.Sequential([

        # TODO: Define the first convolutional layer
        tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),

        # TODO: Define the first max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # TODO: Define the second convolutional layer
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),

        # TODO: Define the second max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # TODO: Define the last Dense layer to output the classification
        # probabilities. Pay attention to the activation needed a probability
        # output
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    return cnn_model


def Alexnet(dim, num_classes, activation='relu', optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=activation,
                               input_shape=(dim, dim, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation=activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def Imagenet(dim, activation='relu', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
    model = tf.keras.models.Sequential([

    ])

    return model

