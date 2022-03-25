import pickle
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import models
import wandb


wandb.init(project="ViT", entity="jschwalb")

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug',  type=bool,
                        help='Keep true unless working on a machine with large memory (75gb+)')
    args = parser.parse_args()

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
    BATCH_SIZE = 64
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

    datasets = [(X[:100], y[:100], 128, "contour analysis", False, False)]
                #(train_rgb, train_labels_rgb, train_rgb.shape[1], "Raw RGB", False, True)]

    test_scores = []

    for item in datasets:
        fig, axes = plt.subplots(2, 1, figsize=(9, 10))

        v = models.ViTransformer(dim=item[2], num_classes=num_classes, num_channel=1)
        v.double()

        x_train = torch.from_numpy(item[0].reshape(item[0].shape[0],1,item[0].shape[1],item[0].shape[2]))
        y_train = torch.from_numpy(item[1])

        x_train_iterable = [x_train[x:x + wandb.config["batch_size"]] for x in range(0, len(x_train), wandb.config["batch_size"])]
        y_train_iterable = [y_train[x:x + wandb.config["batch_size"]] for x in range(0, len(y_train), wandb.config["batch_size"])]

        preds = []
        for x, y in zip(x_train_iterable, y_train_iterable):
            raw_preds = v(x)
            batch_preds = []
            for pred in raw_preds.detach().numpy():
                batch_preds.append(np.argmax(pred))

            preds.append(np.asarray(batch_preds))
            batch_preds = np.asarray(batch_preds)
            batch_acc = accuracy_score(y_true=y.detach().numpy(), y_pred=batch_preds)
            wandb.log({"batch_acc": batch_acc})

        preds = np.asarray(preds)
        acc = accuracy_score(y_true=y_train.detach().numpy(), y_pred=preds)
        wandb.log({"epoch_acc": acc})
        """axes[0].set_xlabel("% of Training Examples")
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
        """




