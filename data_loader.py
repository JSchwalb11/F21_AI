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
        for d in dirs:
            dir1_path = root + '/' + d
            for root1, dirs1, files1 in os.walk(dir1_path, topdown=False):
                for name in files1:
                    fp = root + '/' + d + '/' + name
                    image = np.asarray(resize(fp, ratio=0.5))
                    images.append(image)

    return images

def resize(image_file_path, ratio):
    img = Image.open(image_file_path)
    img_resized = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.ANTIALIAS)
    return img_resized

if __name__ == '__main__':
    data_dir = "C:\Data\Latest"
    data_pickle = "data.pkl"

    data_outfile = open(data_pickle, 'wb')
    train_images = get_train_data(dir_path=data_dir)

    pickle.dump(train_images, data_outfile)
