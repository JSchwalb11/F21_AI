import pickle
import sys
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from pygments.lexers import graphviz
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse
from data_augmentation import pca_reduced_images
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

import models
from time import time as now

import xgboost as xgb
from matplotlib.pylab import rcParams
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from Prototyping import prototype

import wandb

wandb.init(project="my-test-project", entity="jschwalb")



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

    num_classes = 7
    dim = aug_images.shape[1]
    input_type = 'b/w'
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    train_sizes = np.arange(0.1,0.2,0.1)
    max_idx = aug_images.shape[0]

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }

    X, y = aug_images.reshape((aug_images.shape[0], 128, 128)) , aug_labels

    """
    X_pca = pca_reduced_images(X, num_components=None, plot=False)
    X_pca = X_pca.transpose()
    X_pca = X_pca.reshape((X_pca.shape[0], 75, 75))
    X_pca = (255 * (X_pca - np.min(X_pca)) / np.ptp(X_pca)).astype(int)
    """

    #datasets = [(X, 128, "contour analysis"), (X_pca, 75, "PCA Analysis")]
    datasets = [(X, 128, "contour analysis")]
    for item in datasets:
        fig, axes = plt.subplots(2, 1, figsize=(9, 10))

        """classifier_labels = {
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
            train_scores, test_scores = models.plot_learning_curve(classifier,
                                                                   item[0],
                                                                   y,
                                                                   train_sizes=train_sizes,
                                                                   label=label,
                                                                   color=color,
                                                                   axes=axes,
                                                                   datatype=item[2])"""

        cnn_train_scores, cnn_test_scores = models.plot_cnn_learning_curve(images=item[0],
                                                                           labels=y,
                                                                           dim=item[1],
                                                                           train_sizes=train_sizes,
                                                                           num_classes=num_classes,
                                                                           BATCH_SIZE=BATCH_SIZE,
                                                                           EPOCHS=EPOCHS,
                                                                           wandb=None,
                                                                           label="Alexnet (B: {0} E: {1})\n".format(BATCH_SIZE,
                                                                                                                  EPOCHS),
                                                                           axes=axes,
                                                                           color='b')

        axes[0].set_xlabel("% of Training Examples")
        axes[0].set_ylabel("Overall Classification Accuracy")
        axes[0].set_title('Model evaluation - Validation accuracy')

        axes[1].set_xlabel("% of Training Examples")
        axes[1].set_ylabel("Training/Recall Accuracy")
        axes[1].set_title("Model Evaluation - Training Accuracy")
        fig.suptitle("Alexnet trained on {0}".format(item[2]))
        fig.legend(bbox_to_anchor=(1.3, 0.6))
        fig.tight_layout()
        plt.savefig("Alexnet: {0} classes".format(num_classes))
        plt.show()



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


