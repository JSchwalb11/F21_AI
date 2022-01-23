import numpy as np
from PIL import Image
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing



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


def retrieve_contoured_images(train_images, save_dir, input_type, dim):
    filtered_contours, data = image_ops.filter_image_array_contours(train_images, input_type)

    #aug_images1, data1 = data_augmentation.flip_rotate(aug_images, rotate=[0], input_type=input_type, aug_yolo=False, yolo_data=data)
    aug_labels = data_augmentation.generate_labels(filtered_contours, num_classes=6)

    new_images = list()
    new_labels = list()
    new_yolo_labels = list()
    for i, image in enumerate(filtered_contours):
        nan_array = np.isnan(image)
        not_nan_array = ~ nan_array
        arr = image[not_nan_array]
        if len(arr) > 0:
            arr = arr.reshape((128, 128))
            new_images.append(arr)
            new_yolo_labels.append(data[i])
            new_labels.append(aug_labels[i])

    new_yolo_labels = np.array(new_yolo_labels) / dim

    for i in range(0, len(new_images)):
        cls = new_labels[i]
        yolo_data = new_yolo_labels[i]
        img = new_images[i]

        line = str(cls) + " " + \
               str(yolo_data[0]) + " " + \
               str(yolo_data[1]) + " " + \
               str(yolo_data[2]) + " " + \
               str(yolo_data[3])

        filename = save_dir + "{0}\\".format(cls) + str(i)
        PIL.Image.fromarray(img).convert("L").save(filename + ".png")
        with open(filename + ".txt", 'w') as f:
            f.write(line)
            f.close()

    return new_images, new_labels, new_yolo_labels

def pca_reduced_images(train_images):
    new_images = list()
    new_labels = list()
    new_yolo_labels = list()

    b_w_train_images = np.zeros((train_images.shape[0], train_images.shape[1], train_images.shape[2]))
    for i, image in enumerate(train_images):
        rgb_img = PIL.Image.fromarray(image)
        b_w_img = rgb_img.convert('L')
        b_w_train_images[i] = b_w_img

    b_w_train_images = b_w_train_images.reshape((b_w_train_images.shape[0], b_w_train_images.shape[1]*b_w_train_images.shape[2]))

    scaler_model = MinMaxScaler()
    x = scaler_model.fit_transform(b_w_train_images)

    pca = PCA()
    transformed_images = pca.fit_transform(x)
    reduced_pca = np.matmul(x.T, transformed_images)
    reduced_images = reduced_pca.T

    scaler_model1 = MinMaxScaler()
    x = np.trunc(scaler_model1.fit_transform(reduced_images)*255)

    rescaled_images = x.reshape(x.shape[0], 128, 128)


    return rescaled_images.astype('uint8')


if __name__ == '__main__':
    data_dir = "C:\\Data\\Latest"
    #test_data_dir = "C:\\Data\\TestingDir"
    #data_dir = "C:\\Data\\Latest - Old"
    contour_save_dir = "C:\\Data\\Contours\\"
    pca_save_dir = "C:\\Data\\PCA\\"

    input_type = 'b/w'
    #input_type = 'rgb'

    aug_data_pickle = "aug_data.pkl"
    rgb_data_pickle = "rgb_data.pkl"
    aug_labels_pickle = "aug_labels.pkl"
    yolo_labels_pickle = "yolo_labels.pkl"



    train_images = np.asarray(get_train_data(dir_path=data_dir))
    #rgb_images = train_images

    dim = train_images.shape[1]

    new_images, new_labels, new_yolo_labels = retrieve_contoured_images(train_images, dim=dim, save_dir=contour_save_dir, input_type='rgb')
    #reduced_train_images = pca_reduced_images(train_images)
    #new_images, new_labels, new_yolo_labels = retrieve_contoured_images(reduced_train_images, save_dir=contour_save_dir, input_type=input_type)

    aug_data_outfile = open(aug_data_pickle, 'wb')
    aug_labels_outfile = open(aug_labels_pickle, 'wb')
    #rgb_data_outfile = open(rgb_data_pickle, 'wb')
    yolo_labels_outfile = open(yolo_labels_pickle, 'wb')

    pickle.dump(new_images, aug_data_outfile)
    pickle.dump(new_labels, aug_labels_outfile)
    pickle.dump(new_yolo_labels, yolo_labels_outfile)
    #pickle.dump(rgb_images, rgb_data_outfile)

    #pickle.dump(edge_images, edge_outfile)
