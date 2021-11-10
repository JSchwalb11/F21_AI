import pickle
import sys
import cv2
import numpy as np
import tensorflow as tf
from pygments.lexers import graphviz
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from time import time as now
import argparse
import shap
from sklearn import tree
from sklearn import metrics
import graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler




import image_ops
from PIL import Image
import models
import data_loader
import math


def plot_learning_curve(classifier, X, y, steps=10, train_sizes=np.arange(0.1,0.6,0.1, dtype=np.float32), label="",
                        color='r', axes=None):
    estimator=Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        print("Training {0} model".format(label))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42)
        #y_train_encoded = to_categorical(y_train)
        #y_test_encoded = to_categorical(y_test)

        """if label == "Alexnet":
            y_train = y_train_encoded
            y_test = y_test_encoded
            estimator.fit(X_train, y_train)#,
                          #validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)
        else:"""
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

        estimator.fit(X_train, y_train)

        train_score = estimator.score(X_train, y_train) * 100
        test_score = estimator.score(X_test, y_test) * 100
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("Train Score: {0}\nTest Score: {1}\n".format(train_score, test_score))


    if axes is None:
        _, axes= plt.subplots(2)

    axes[0].plot(train_sizes, test_scores, "o-", color=color, label=label)
    axes[1].plot(train_sizes, train_scores, "o-", color=color, label=label)

    print("Training Accuracy of ", label, ": ", train_scores[-1],"%")
    print("Testing Accuracy of ", label, ": ", test_scores[-1], "%")
    print()

    return train_scores, test_scores

def plot_cnn_learning_curve(images, labels, dim, num_classes, BATCH_SIZE, EPOCHS, train_sizes=np.arange(0.1,0.6,0.1), label="", color='r', axes=None):
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, train_size=train_size, random_state=42)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        #train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
        #test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

        model = models.Alexnet_bw_input(dim=dim, num_classes=num_classes, SHAP=True)

        start = now()
        history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)
        print("Total training time in %.3f" % (now() - start))

        #background = images[np.random.choice(images.shape[0], 100, replace=False)]
        #explainer = shap.DeepExplainer(model, background)
        #shap_values = explainer.shap_values(test_images[0:4])
        #shap.image_plot(shap_values, -test_images[0:4])

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug',  type=bool,
                        help='Keep true unless working on a machine with large memory (75gb+)')
    args = parser.parse_args()

    aug_data_pickle = "aug_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"
    yolo_labels_pickle = "yolo_labels.pkl"

    aug_images_file = open(aug_data_pickle, 'rb')
    aug_labels_file = open(aug_labels_pickle, 'rb')
    yolo_labels_pickle = open(yolo_labels_pickle, 'rb')


    aug_images = pickle.load(aug_images_file)
    aug_labels = pickle.load(aug_labels_file)
    yolo_labels = pickle.load(yolo_labels_pickle)

    aug_images = np.asarray(aug_images)
    aug_labels = np.asarray(aug_labels)

    if (len(aug_images) == len(aug_labels)):
        print("Lengths match")
    else:
        print("Lengths do not match, check data loader")
        sys.exit(0)

    num_classes = 4
    dim = aug_images.shape[1]
    input_type = 'b/w'
    BATCH_SIZE = 30
    EPOCHS = 15

    train_images, test_images, train_labels, test_labels = train_test_split( \
        aug_images[:, :dim, :dim], aug_labels, test_size=0.25, random_state=42)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    """if input_type == 'rgb':
        model = models.Alexnet(dim=dim, num_classes=num_classes)
    else:
        train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
        test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
        #model = models.Alexnet_bw_input(dim=dim, num_classes=num_classes, SHAP=True)
    """
    X, y = aug_images, aug_labels

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    classifier_labels = {
        "SVM - Poly": (svm.SVC(kernel='poly', random_state=1), "yellow"),
        "SVM - RBF": (svm.SVC(kernel='rbf', random_state=1), "orange"),
        "kNN": (KNeighborsClassifier(n_neighbors=5), "purple"),
        "Gaussian Naive Bayes": (GaussianNB(), "lime"),
        "LDA": (LinearDiscriminantAnalysis(), "red"),
        "DTree": (tree.DecisionTreeClassifier(), "cyan")#,
        #"Alexnet": (models.Alexnet_bw_input(dim=dim, num_classes=num_classes, SHAP=True), "blue",
        #            BATCH_SIZE, EPOCHS)
    }
    for label in classifier_labels:
        classifier = classifier_labels[label][0]
        color = classifier_labels[label][1]
        train_scores, test_scores = plot_learning_curve(classifier, X, y, label=label, color=color, axes=axes)

    cnn_train_scores, cnn_test_scores = plot_cnn_learning_curve(images=X,
                                                                labels=y,
                                                                dim=dim,
                                                                num_classes=num_classes,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                EPOCHS=EPOCHS,
                                                                label="Alexnet (B: {0}/E: {1}".format(BATCH_SIZE, EPOCHS),
                                                                axes=axes,
                                                                color='olive')


    axes[0].set_xlabel("% of Training Examples")
    axes[0].set_ylabel("Overall Classification Accuracy")
    axes[0].set_title('Model evaluation - Validation accuracy')
    axes[0].legend()

    axes[1].set_xlabel("% of Training Examples")
    axes[1].set_ylabel("Training/Recall Accuracy")
    axes[1].set_title("Model Evaluation - Training Accuracy")
    axes[1].legend()
    plt.show()


    """    
    import os
    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
    X1 = X.copy()
    X1 = X1.reshape((X1.shape[0], X1.shape[1] * X1.shape[2] * X1.shape[3]))
    x1_test = test_images.copy()
    x1_test = x1_test.reshape((x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2] * x1_test.shape[3]))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X1, y)
    y_pred = clf.predict(x1_test)
    print("Decision Tree Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    dot_data = tree.export_graphviz(clf, out_file=None)
    
    graph = graphviz.Source(dot_data)
    graph.render("Detection")
    """
    """model = models.Alexnet_bw_input(dim=dim, num_classes=num_classes, SHAP=True)
    BATCH_SIZE = 30
    EPOCHS = 15
    train_scores = []
    test_scores = []
    train_sizes = np.arange(0.1, 0.6, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p = np.random.permutation(len(train_images))
    #with tf.device("/device:cpu:0"):
    cnn_train_scores, cnn_test_scores = plot_cnn_learning_curve(images=X, labels=y, dim=dim,
                                                                num_classes=num_classes, BATCH_SIZE=BATCH_SIZE,
                                                                EPOCHS=EPOCHS, label="Alexnet", axes=axes,
                                                                color='olive')

    axes[0].set_xlabel("% of Training Examples")
    axes[0].set_ylabel("Overall Classification Accuracy")
    axes[0].set_title('Model evaluation - validation accuracy')
    axes[0].legend()

    axes[1].set_xlabel("% of Training Examples")
    axes[1].set_ylabel("Training/Recall Accuracy")
    axes[1].set_title("Model Evaluation - Training Accuracy")
    axes[1].legend()
    plt.show()"""

    """# plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()"""


