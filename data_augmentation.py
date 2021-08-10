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

