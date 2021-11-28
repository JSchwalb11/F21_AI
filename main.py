import pickle
import sys
import cv2
import numpy as np
import tensorflow as tf
from pygments.lexers import graphviz
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import models
from time import time as now

import xgboost as xgb
from matplotlib.pylab import rcParams
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug',  type=bool,
                        help='Keep true unless working on a machine with large memory (75gb+)')
    args = parser.parse_args()

    aug_data_pickle = "aug_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"
    yolo_labels_pickle = "yolo_labels.pkl"
    rgb_data_pickle = "rgb_data.pkl"

    aug_images_file = open(aug_data_pickle, 'rb')
    aug_labels_file = open(aug_labels_pickle, 'rb')
    yolo_labels_pickle = open(yolo_labels_pickle, 'rb')
    rgb_images_file = open(rgb_data_pickle, 'rb')


    aug_images = pickle.load(aug_images_file)
    aug_labels = pickle.load(aug_labels_file)
    yolo_labels = pickle.load(yolo_labels_pickle)
    rgb_images = pickle.load(rgb_images_file)

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

    X, y = aug_images, aug_labels
    df = pd.DataFrame(aug_labels)#, columns={'Hostage', 'Trailer', 'House', 'Fountain'})
    fig, axes = plt.subplots(1,2,figsize=(12,5))

    """
    classifier_labels = {
        "SVM - Poly": (svm.SVC(kernel='poly', random_state=1), "yellow"),
        "SVM - RBF": (svm.SVC(kernel='rbf', random_state=1), "orange"),
        "kNN": (KNeighborsClassifier(n_neighbors=5), "purple"),
        "Gaussian Naive Bayes": (GaussianNB(), "lime"),
        "LDA": (LinearDiscriminantAnalysis(), "red"),
        "DTree": (tree.DecisionTreeClassifier(), "cyan")#,
    }
    for label in classifier_labels:
        classifier = classifier_labels[label][0]
        color = classifier_labels[label][1]
        train_scores, test_scores = models.plot_learning_curve(classifier, X, y, label=label, color=color, axes=axes)
    """
    cnn_train_scores, cnn_test_scores = models.plot_cnn_learning_curve(images=X,
                                                                       labels=y,
                                                                       dim=dim,
                                                                       train_sizes=np.arange(0.1, 0.6, 0.1),
                                                                       num_classes=num_classes,
                                                                       BATCH_SIZE=BATCH_SIZE,
                                                                       EPOCHS=EPOCHS,
                                                                       label="Alexnet (B: {0} E: {1})\n(No preprocessing)".format(BATCH_SIZE,
                                                                                                              EPOCHS),
                                                                       axes=axes,
                                                                       color='b')

    axes[0].set_xlabel("% of Training Examples")
    axes[0].set_ylabel("Overall Classification Accuracy")
    axes[0].set_title('Model evaluation - Validation accuracy')
    axes[0].legend()

    axes[1].set_xlabel("% of Training Examples")
    axes[1].set_ylabel("Training/Recall Accuracy")
    axes[1].set_title("Model Evaluation - Training Accuracy")
    fig.suptitle("Alexnet (B: {0} E: {1}) (No preprocessing)".format(BATCH_SIZE, EPOCHS))
    axes[1].legend()
    plt.savefig("Alexnet: No preprocessing (Raw RGB)")
    plt.show()
    """
    # Run below tonight
    colors = ['r', 'b', 'g', 'y', 'p', 'black', 'olive']
    min_batch = 5
    max_batch = 50
    min_epoch = 5
    max_epoch = 50
    step = 5
    train_sizes = np.arange(0.1, 0.6, 0.1)
    batch_sizes = np.arange(min_batch, max_batch+1, step)
    epoch_sizes = np.arange(min_epoch, max_epoch+1, step)
    hyperparams = {'batch': 0, 'epoch': 0, 'val_acc': 0}
    start = now()
    print("Starting Search")
    for i, EPOCH in enumerate(epoch_sizes):
        for j, BATCH in enumerate(batch_sizes):
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

            cnn_train_scores, cnn_test_scores = models.plot_cnn_learning_curve(images=X,
                                                                    labels=y,
                                                                    dim=dim,
                                                                    train_sizes=train_sizes,
                                                                    num_classes=num_classes,
                                                                    BATCH_SIZE=BATCH,
                                                                    EPOCHS=EPOCH,
                                                                    label="Alexnet (B: {0} E: {1})".format(BATCH, EPOCH),                                                                axes=axes,
                                                                    color=colors[i])
            best_acc = np.asarray(cnn_test_scores).max()

            if best_acc > hyperparams['val_acc']:
                hyperparams['batch'] = BATCH
                hyperparams['epoch'] = EPOCH
                hyperparams['val_acc'] = best_acc

            axes[0].set_xlabel("% of Training Examples")
            axes[0].set_ylabel("Overall Classification Accuracy")
            axes[0].set_title('Model evaluation - Validation accuracy')
            axes[0].legend()

            axes[1].set_xlabel("% of Training Examples")
            axes[1].set_ylabel("Training/Recall Accuracy")
            axes[1].set_title("Model Evaluation - Training Accuracy")
            fig.suptitle("Hyperparameter Search\nBATCH_SIZE: {0}, EPOCHS: {1}".format(BATCH, EPOCH))
            axes[1].legend()
            plt.savefig("Hyperparameter Search BATCH_SIZE-{0} EPOCHS-{1}".format(BATCH, EPOCH))
    end = now()
    time_elapsed = end-start
    print("Searched for hyperparameters within bounds:" +
          "\nMin Batch Size: {0}\nMax Batch Size: {1}\nMin Epoch Size: {2}\nMax Epoch Size: {3}\nStep: {4}".format(
              min_batch, max_batch, min_epoch, max_epoch, step))
    print("\nFound: {0}\n".format(hyperparams))
    print("Time elapsed: {0}".format(time_elapsed))
    """
    # Run above tonight
    """X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, train_size=0.3, random_state=42)
    dtrain = xgb.DMatrix(train_images, label=train_labels)
    dtest = xgb.DMatrix(test_images, label=test_labels)
    param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'num_class': num_classes,
             'figure.figsize': (80, 50), 'nthread': 4, 'eval_metric': 'auc'}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)

    bst.save_model('0001.model')

    # dump model
    bst.dump_model('dump.raw.txt')
    # dump model with feature map
    #bst.dump_model('dump.raw.txt', 'featmap.txt')
    xgb.plot_importance(bst)

    xgb.plot_tree(bst, num_trees=2)

    plt.show()"""


