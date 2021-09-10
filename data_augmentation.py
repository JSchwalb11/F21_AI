import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def imSave(img, filename):
    """helper function to show data augmentation
    :param img: path of the image
    :param transform: data augmentation technique to apply"""

    im = PIL.Image.fromarray(img, 'RGB')
    im.save(filename + ".png", "PNG")

def transform(image, flip_x = False, flip_y = False, rotate = None):
    tmp = PIL.Image.fromarray(image, 'RGB')

    if flip_x == True:
        tmp = tmp.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    if flip_y == True:
        tmp = tmp.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    if rotate == 90:
        tmp = tmp.transpose(PIL.Image.ROTATE_90)
    elif rotate == 180:
        tmp = tmp.transpose(PIL.Image.ROTATE_180)
    elif rotate == 270:
        tmp = tmp.transpose(PIL.Image.ROTATE_270)
    #else:
        #print("This message doesnt help")

    return np.asarray(tmp)

def flip_rotate(images, rotate=[0]):
    flip_x = [False, True]
    flip_y = [False, True]
    # rotate = [0, 90, 180, 270]
    # rotate = [0, 90]
    # rotate = [0]

    permutations = len(flip_x) * len(flip_y) * len(rotate)

    aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2], images.shape[3]),
                          dtype=np.uint8)

    for i, image in enumerate(images):
        for i_j, j in enumerate(flip_x):
            for i_k, k in enumerate(flip_y):
                for i_r, r in enumerate(rotate):
                    offset = i_j + i_k + i_r
                    start = i * permutations
                    end = start + offset
                    aug_images[start + offset] = transform(image, flip_x=j, flip_y=k, rotate=r)

    return aug_images

def generate_labels(aug_images, num_classes):
    aug_labels = np.zeros(len(aug_images), dtype=np.uint8)
    slice = len(aug_images) // num_classes

    for i in range(0, num_classes):
        aug_labels[slice * i: slice * (i + 1)] = i

    return aug_labels


