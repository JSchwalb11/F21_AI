from PIL import Image
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split




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


def retrieve_contoured_images(train_images, save_dir, input_type, dim, num_classes):
    filtered_contours, data = image_ops.filter_image_array_contours(train_images, input_type)

    #aug_images1, data1 = data_augmentation.flip_rotate(aug_images, rotate=[0], input_type=input_type, aug_yolo=False, yolo_data=data)
    aug_labels = data_augmentation.generate_labels(filtered_contours, num_classes=num_classes)

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




if __name__ == '__main__':
    data_dir = "C:\\Data\\Latest"
    contour_save_dir = "C:\\Data\\Contours\\"
    pca_save_dir = "C:\\Data\\PCA\\"

    input_type = 'b/w'
    #input_type = 'rgb'

    num_classes = 7

    for i in range(0, num_classes):
        path = os.path.join(contour_save_dir, str(i))

        try:
            os.mkdir(path)
            print("Created Directory")
        except OSError as error:
            print("Directory Already Exists")

    train_images = np.asarray(get_train_data(dir_path=data_dir))
    train_labels = data_augmentation.generate_labels(train_images, num_classes=num_classes)

    dim = train_images.shape[1]

    contour_images, contour_labels, yolo_labels = retrieve_contoured_images(train_images,
                                                                        dim=dim,
                                                                        save_dir=contour_save_dir,
                                                                        input_type='rgb',
                                                                        num_classes=num_classes)
    tmp = 0
    for i, label in enumerate(train_labels):
        if label != tmp:
            print("New class at idx {0}".format(i))
            tmp = label

    """X_pca, img_dim = data_augmentation.pca_reduced_images(np.asarray(contour_images),
                                                          num_components=None,
                                                          plot=False)"""

    """train_contour, test_contour, train_labels_contour, test_labels_contour = train_test_split(
        np.asarray(contour_images), np.asarray(contour_labels), train_size=0.1, random_state=42)

    train_rgb, test_rgb, train_labels_rgb, test_labels_rgb = train_test_split(
        np.asarray(train_images), np.asarray(train_labels), train_size=0.1, random_state=42)"""

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

    pickle_outfiles = []
    for file in pickle_files:
        pickle_outfiles.append(open(file, 'wb'))

    data_to_write = [train_contour,
                     train_labels_contour,
                     test_contour,
                     test_labels_contour,
                     train_rgb,
                     train_labels_rgb,
                     test_rgb,
                     test_labels_rgb,
                     X_pca,
                     yolo_labels]

    for i, outfile in enumerate(pickle_outfiles):
        pickle.dump(data_to_write[i], outfile)"""


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
    
    aug_data_outfile = open(aug_data_pickle, 'wb')
    aug_labels_outfile = open(aug_labels_pickle, 'wb')
    aug_test_data_outfile = open(aug_test_data_pickle, 'wb')
    aug_test_labels_outfile = open(aug_test_labels_pickle, 'wb')

    rgb_data_outfile = open(rgb_data_pickle, 'wb')
    rgb_labels_outfile = open(rgb_labels_pickle, 'wb')
    rgb_test_data_outfile = open(rgb_test_data_pickle, 'wb')
    rgb_test_labels_outfile = open(rgb_test_labels_pickle, 'wb')

    pca_data_outfile = open(pca_data_pickle, 'wb')

    yolo_labels_outfile = open(yolo_labels_pickle, 'wb')

    pickle.dump(contour_images, aug_data_outfile)
    pickle.dump(contour_labels, aug_labels_outfile)
    #pickle.dump(test_contour, aug_test_data_outfile)
    #pickle.dump(test_labels_contour, aug_test_labels_outfile)

    pickle.dump(train_images, rgb_data_outfile)
    pickle.dump(train_labels, rgb_labels_outfile)
    #pickle.dump(test_rgb, rgb_test_data_outfile)
    #pickle.dump(test_labels_rgb, rgb_test_labels_outfile)

    #pickle.dump(X_pca, pca_data_outfile)

    pickle.dump(yolo_labels, yolo_labels_outfile)


