from PIL import Image
import os
import numpy as np
import pickle

def get_train_data(dir_path):
    """

    :param file_path: path to data
    :return: numpy array of images
    """
    images = []

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            fp = root + '/' + name
            image = np.asarray(resize(fp, ratio=0.5))
            images.append(image)

    return images

def get_test_data(dir_path):
    images = []

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            fp = root + '/' + name
            image = np.asarray(resize(fp, ratio=0.5))
            images.append(image)

    return images

def resize(image_file_path, ratio):
    img = Image.open(image_file_path)
    img_resized = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.ANTIALIAS)
    return img_resized

if __name__ == '__main__':
    train_fp = "C:\Data\Snapshots"
    test_fp = "C:\Data\Hostage Dataset"

    train_pickle = "train_data"
    test_pickle = "test_data"

    train_outfile = open(train_pickle, 'wb')
    test_outfile = open(test_pickle, 'wb')



    train_images = get_train_data(dir_path=train_fp)
    test_images = get_test_data(dir_path=test_fp)

    pickle.dump(train_images, train_outfile)
    pickle.dump(test_images, test_outfile)