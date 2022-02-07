import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from time import time as now
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import shap
import sys
from wandb.keras import WandbCallback

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
                               input_shape=(dim, dim, 3)), # for use with rgb images
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

def Alexnet_bw_input(dim, num_classes, activation='relu', optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], SHAP=False, small_input=False):
    if SHAP == True:
        tf.compat.v1.disable_v2_behavior()

    if small_input == True:
        pool_size = (2, 2)
    else:
        pool_size = (3, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=activation,
                               input_shape=(dim, dim, 1)),# for use with b/w images
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=(2, 2)),
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

def plot_cnn_learning_curve(images, labels, dim, num_classes, BATCH_SIZE, EPOCHS, train_sizes=np.arange(0.1,0.6,0.1), label="", color='r', axes=None, small_input=False):
    #images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))
    images = images.reshape((images.shape[0], dim, dim))

    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, train_size=train_size, random_state=42)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        print("Train size {0} images".format(len(train_images)))
        print("Train size {0} bytes".format(sys.getsizeof(train_images)))
        print("Test size {0} images".format(len(test_images)))
        print("Test size {0} bytes".format(sys.getsizeof(test_images)))
        model = Alexnet_bw_input(dim=dim, num_classes=num_classes, SHAP=False, small_input=small_input)

        #model = Alexnet(dim=dim, num_classes=num_classes)

        start = now()
        history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                            batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[WandbCallback()])
        print("Total training time in %.3f" % (now() - start))

        #background = images[np.random.choice(images.shape[0], 100, replace=False)]
        #explainer = shap.DeepExplainer(model, background)
        #shap_values = explainer.shap_values(test_images[0:4])
        #shap.image_plot(shap_values, -test_images[0:4])
        #plt.savefig("Shap Analysis")

        with tf.device("/device:cpu:0"):
            _, train_acc = model.evaluate(train_images, train_labels, verbose=0)

            _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
            train_scores.append(train_acc * 100)
            test_scores.append(test_acc * 100)
            print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    if axes is None:
        _, axes = plt.subplots(2)

    axes[0].plot(train_sizes, test_scores, "o-", color=color, label=label)
    axes[1].plot(train_sizes, train_scores, "o-", color=color, label=label)

    return train_scores, test_scores

def plot_learning_curve(classifier, X, y, steps=10, train_sizes=np.arange(0.1,0.6,0.1, dtype=np.float32), label="",
                        color='r', axes=None, datatype=None):
    estimator=Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        print("Training {0} model".format(label))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

        estimator.fit(X_train, y_train)

        train_score = estimator.score(X_train, y_train) * 100
        test_score = estimator.score(X_test, y_test) * 100
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("Train Score: {0}\nTest Score: {1}\n".format(train_score, test_score))

    if axes is None:
        _, axes = plt.subplots(2)


    axes[0].plot(train_sizes, test_scores, "o-", color=color, label=label)
    axes[1].plot(train_sizes, train_scores, "o-", color=color, label=label)

    print("Training Accuracy of ", label, ": ", train_scores[-1],"%")
    print("Testing Accuracy of ", label, ": ", test_scores[-1], "%")
    print()

    return train_scores, test_scores
