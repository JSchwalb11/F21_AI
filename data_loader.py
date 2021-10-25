import numpy
from PIL import Image
import os
import numpy as np
import pandas as pd
import pickle

import data_augmentation
import feature_extraction
import PIL
import image_ops


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

def review(train_images):
    return 0


if __name__ == '__main__':
    data_dir = "C:\\Data\\Latest"
    save_dir = "C:\\Data\\Contours\\"
    #input_type = 'b/w'
    input_type = 'rgb'

    aug_data_pickle = "aug_data.pkl"
    rgb_data_pickle = "rgb_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"


    aug_data_outfile = open(aug_data_pickle, 'wb')
    aug_labels_outfile = open(aug_labels_pickle, 'wb')
    rgb_data_outfile = open(rgb_data_pickle, 'wb')

    train_images = np.asarray(get_train_data(dir_path=data_dir))
    #sanity_checked_images, images_for_review = review(train_images)

    if input_type == 'rgb':
        #aug_images1 = data_augmentation.flip_rotate(train_images, rotate=[0], input_type=input_type)
        #rgb_images = data_augmentation.flip_rotate(train_images, rotate=[0], input_type=input_type)
        aug_images, data = image_ops.filter_image_array_contours(train_images)
        aug_labels = data_augmentation.generate_labels(aug_images, num_classes=4)
        a = list()
        #a = numpy.zeros((aug_images.shape[0], aug_images.shape[1], aug_images[2]), dtype=aug_images.dtype)
        for i, image in enumerate(aug_images):
            nan_array = np.isnan(image)
            not_nan_array = ~ nan_array
            arr = image[not_nan_array]
            if len(arr) > 0:
                arr = arr.reshape((128, 128))
                a.append((aug_labels[i], data[i], arr))

        for i, arr in enumerate(a):
            cls, data, img = arr[0], arr[1], arr[2]

            line = str(aug_labels[i]) + " " + \
                   str(data[0]) + " " + \
                   str(data[1]) + " " + \
                   str(data[2]) + " " + \
                   str(data[3])
            filename = save_dir + "{0}\\".format(cls) + str(i)
            PIL.Image.fromarray(img).convert("L").save(filename + ".png")
            with open(filename + ".txt", 'w') as f:
                f.write(line)
                f.close()
    """
    else:
        edge_extraction = image_ops.extract_edges(train_images)
        aug_images = data_augmentation.flip_rotate(edge_extraction, rotate=[0], input_type=input_type)
        aug_labels = data_augmentation.generate_labels(aug_images, num_classes=4)





    pickle.dump(a[1], aug_data_outfile)
    pickle.dump(aug_labels, aug_labels_outfile)

    #pickle.dump(rgb_images, rgb_data_outfile)
    #pickle.dump(edge_images, edge_outfile)
    """
