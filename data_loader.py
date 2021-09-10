from PIL import Image
import os
import numpy as np
import pickle

import data_augmentation
import feature_extraction
import time

def get_train_data(dir_path):
    """

    :param file_path: path to data
    :return: numpy array of images
    """
    images = []

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for d in dirs:
            dir1_path = root + '/' + d
            for root1, dirs1, files1 in os.walk(dir1_path, topdown=False):
                for name in files1:
                    fp = root + '/' + d + '/' + name
                    print(fp)
                    image = np.asarray(resize(fp, ratio=0.25), dtype=np.uint8)
                    images.append(image)

    return images

def resize(image_file_path, ratio):
    img = Image.open(image_file_path)
    img_resized = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.ANTIALIAS)
    return img_resized

def get_edge_data(img_arr):
    edge_images = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2], img_arr.shape[3]),
                           dtype=np.uint8)

    for i, image in enumerate(img_arr):
        edge_images[i] = feature_extraction.get_edge(image)
        print("Got edges for image {0}".format(i))

    return edge_images

if __name__ == '__main__':
    data_dir = "C:\Data\Latest"

    aug_data_pickle = "aug_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"

    aug_data_outfile = open(aug_data_pickle, 'wb')
    aug_labels_outfile = open(aug_labels_pickle, 'wb')

    train_images = np.asarray(get_train_data(dir_path=data_dir))
    aug_images = data_augmentation.flip_rotate(train_images, rotate=[0])
    aug_labels = data_augmentation.generate_labels(aug_images, num_classes=4)

    pickle.dump(aug_images, aug_data_outfile)
    pickle.dump(aug_labels, aug_labels_outfile)
    #pickle.dump(edge_images, edge_outfile)
