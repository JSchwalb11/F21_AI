import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import models
from Prototyping import prototype
import wandb
import math
import pandas as pd

wandb.init(project="my-test-project", entity="jschwalb")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug',  type=bool,
                        help='Keep true unless working on a machine with large memory (75gb+)')
    args = parser.parse_args()

    """pickle_files = ["aug_data.pkl",
                    "aug_labels.pkl",
                    "aug_test_data.pkl",
                    "aug_test_labels.pkl",
                    "rgb_data.pkl",
                    "rgb_labels.pkl",
                    "rgb_test_data.pkl",
                    "rgb_test_labels.pkl",
                    "pca_data.pkl",
                    "yolo_labels.pkl"]

    variables = []
    for i, file in enumerate(pickle_files):
        #pickle_infiles.append(open(file, 'rb'))
        variables.append(pickle.load(open(file, 'rb')))

    train_contour = variables[0]
    train_labels_contour = variables[1]
    test_contour = variables[2]
    test_labels_contour = variables[3]

    train_rgb = variables[4]
    train_labels_rgb = variables[5]
    test_rgb = variables[6]
    test_labels_rgb = variables[7]

    X_pca = variables[8]
    yolo_labels = variables[9]"""


    aug_data_pickle = "aug_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"
    aug_test_data_pickle = "aug_test_data.pkl"
    aug_test_labels_pickle = "aug_test_labels.pkl"

    rgb_data_pickle = "rgb_data.pkl"
    rgb_labels_pickle = "rgb_labels.pkl"
    rgb_test_data_pickle = "rgb_test_data.pkl"
    rgb_test_labels_pickle = "rgb_test_labels.pkl"

    pca_data_pickle = "pca_data.pkl"
    pca_test_data_pickle = "pca_test_data.pkl"

    yolo_labels_pickle = "yolo_labels.pkl"

    aug_data_file = open(aug_data_pickle, 'rb')
    aug_labels_file = open(aug_labels_pickle, 'rb')
    aug_test_data_file = open(aug_test_data_pickle, 'rb')
    aug_test_labels_file = open(aug_test_labels_pickle, 'rb')

    rgb_data_file = open(rgb_data_pickle, 'rb')
    rgb_labels_file = open(rgb_labels_pickle, 'rb')
    rgb_test_data_file = open(rgb_test_data_pickle, 'rb')
    rgb_test_labels_file = open(rgb_test_labels_pickle, 'rb')

    pca_data_file = open(pca_data_pickle, 'rb')

    yolo_labels_file = open(yolo_labels_pickle, 'rb')


    train_contour = pickle.load(aug_data_file)
    train_labels_contour = pickle.load(aug_labels_file)
    #aug_test_images = pickle.load(aug_test_data_file)
    #aug_test_labels = pickle.load(aug_test_labels_file)

    train_rgb = pickle.load(rgb_data_file)
    train_labels_rgb = pickle.load(rgb_labels_file)
    #rgb_test_images = pickle.load(rgb_test_data_file)
    #rgb_test_labels = pickle.load(rgb_test_labels_file)

    #X_pca = pickle.load(pca_data_file)


    #img_dim = int(math.sqrt(X_pca.shape[1]).__trunc__())
    #print("PCA Img Dimensions: {0}x{1}".format(img_dim, img_dim))

    train_contour = np.asarray(train_contour)
    train_labels_contour = np.asarray(train_labels_contour)
    train_rgb = np.asarray(train_rgb)
    train_labels_rgb = np.asarray(train_labels_rgb)

    if (len(train_contour) == len(train_labels_contour)):
        print("Lengths match")
    else:
        print("Lengths do not match, check data loader")
        sys.exit(0)

    num_classes = 7
    dim = train_contour.shape[1]
    input_type = 'b/w'
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.001
    train_sizes = np.arange(0.1,0.6,0.1)
    max_idx = train_contour.shape[0]

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }

    X, y = train_contour.reshape((train_contour.shape[0], 128, 128)) , train_labels_contour


    #X_pca = (255 * (X_pca - np.min(X_pca)) / np.ptp(X_pca)).astype(int)

    datasets = [(X, y, 128, "contour analysis", False, False),
                #(X_pca, y, img_dim, "contour + PCA Analysis", False, False),
                (train_rgb, train_labels_rgb, train_rgb.shape[1], "Raw RGB", False, True)]

    test_scores = []

    for item in datasets:
        fig, axes = plt.subplots(2, 1, figsize=(9, 10))

        classifier_labels = {
            "SVM - Poly": (svm.SVC(kernel='poly', random_state=1), "yellow"),
            #"SVM - RBF": (svm.SVC(kernel='rbf', random_state=1), "orange"),
            #"kNN": (KNeighborsClassifier(n_neighbors=5), "purple"),
            "Gaussian Naive Bayes": (GaussianNB(), "lime"),
            #"LDA": (LinearDiscriminantAnalysis(), "red"),
            #"DTree": (tree.DecisionTreeClassifier(), "cyan")#,
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
                                                                   datatype=item[2])

        cnn_train_scores, cnn_test_scores = models.plot_cnn_learning_curve(images=item[0],
                                                                           labels=item[1],
                                                                           dim=item[2],
                                                                           train_sizes=train_sizes,
                                                                           num_classes=num_classes,
                                                                           BATCH_SIZE=BATCH_SIZE,
                                                                           EPOCHS=EPOCHS,
                                                                           label="Alexnet (B: {0} E: {1})\n".format(BATCH_SIZE,
                                                                                                                  EPOCHS),
                                                                           axes=axes,
                                                                           color='b',
                                                                           small_input=item[4],
                                                                           rgb=item[5])

        axes[0].set_xlabel("% of Training Examples")
        axes[0].set_ylabel("Overall Classification Accuracy")
        axes[0].set_title('Model evaluation - Validation accuracy')

        axes[1].set_xlabel("% of Training Examples")
        axes[1].set_ylabel("Training/Recall Accuracy")
        axes[1].set_title("Model Evaluation - Training Accuracy")
        fig.suptitle("Alexnet trained on {0}".format(item[3]))
        fig.legend(bbox_to_anchor=(1.3, 0.6))
        fig.tight_layout()
        plt.savefig("Alexnet: {0} classes".format(num_classes))
        plt.show()

        test_scores.append(cnn_test_scores)





