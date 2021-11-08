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

def transform(image, flip_x = False, flip_y = False, rotate = None, input_type='rgb'):
    if input_type == 'rgb':
        tmp = PIL.Image.fromarray(image, 'RGB')
    else:
        tmp = PIL.Image.fromarray(image, 'L')

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

def transform_yolo_coords(input, flip_x = False, flip_y = False, rotate = None):
    coords = input.copy()
    if coords.all() == None:
        return None
    #else:
    #    return None

    if flip_x == True:
        coords[0] = coords[0]
        coords[1] = 0 - coords[1]
        coords[2] = coords[2]
        coords[3] = 0 - coords[3]

    if flip_y == True:
        coords[0] = 0 - coords[0]
        coords[1] = coords[1]
        coords[2] = 0 - coords[2]
        coords[3] = coords[3]

    if rotate == 90:
        coords[0] = 0 - coords[1]
        coords[1] = coords[0]
        coords[2] = 0 - coords[3]
        coords[3] = coords[2]

    elif rotate == 180:
        coords[0] = 0 - coords[0]
        coords[1] = 0 - coords[1]
        coords[2] = 0 - coords[2]
        coords[3] = 0 - coords[3]

    elif rotate == 270:
        coords[0] = coords[1]
        coords[1] = 0 - coords[0]
        coords[2] = coords[3]
        coords[3] = 0 - coords[2]

    return coords

def flip_rotate(images, rotate=[0], input_type='rgb'):
    flip_x = [False, True]
    flip_y = [False, True]
    # rotate = [0, 90, 180, 270]
    # rotate = [0, 90]
    # rotate = [0]

    permutations = len(flip_x) * len(flip_y) * len(rotate)

    if input_type =='b/w':
        aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2]),
                              dtype=np.uint8)
    else:
        aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2], images.shape[3]),
                          dtype=np.uint8)

    for i, image in enumerate(images):
        for i_j, j in enumerate(flip_x):
            for i_k, k in enumerate(flip_y):
                for i_r, r in enumerate(rotate):
                    offset = i_j + i_k + i_r
                    start = i * permutations
                    end = start + offset
                    aug_images[start + offset] = transform(image, flip_x=j, flip_y=k, rotate=r, input_type=input_type)

    return aug_images

def flip_rotate(images, rotate=[0], input_type='rgb', aug_yolo=False, yolo_data=None):
    """

    :param images:
    :param rotate:
    :param input_type:
    :param aug_yolo:
    :param yolo_data:
    :return:

    1. No change
    2. Flip_x = True
    3. Flip_y = True
    4. Flip_x==Flip_y==True
    5. Rotate_90, No flip
    6. Rotate_90, Flip_x = True
    7. Rotate_90, Flip_y = True
    8. Rotate_90, Flip_x==Flip_y==True
    9. Rotate_180, No flip
    10. Rotate_180, Flip_x = True
    11. Rotate_180, Flip_y = True
    12. Rotate_180, Flip_x==Flip_y==True
    13. Rotate_270, No flip
    14. Rotate_270, Flip_x = True
    15. Rotate_270, Flip_y = True
    16. Rotate_270, Flip_x==Flip_y==True
    """

    flip_x = [False, True]
    flip_y = [False, True]
    # rotate = [0, 90, 180, 270]
    # rotate = [0, 90]
    # rotate = [0]

    permutations = len(flip_x) * len(flip_y) * len(rotate)

    if input_type =='b/w':
        aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2]),
                              dtype=np.uint8)
        aug_yolo = np.zeros((yolo_data.shape[0] * permutations, yolo_data[0].shape[0]))
    else:
        aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2], images.shape[3]),
                          dtype=np.uint8)

    for i, image in enumerate(images):
        print("i: {0}".format(i))
        for i_j, j in enumerate(flip_x):
            for i_k, k in enumerate(flip_y):
                for i_r, r in enumerate(rotate):
                    offset = i_j + i_k + i_r
                    start = i * permutations
                    end = start + offset
                    aug_images[start + offset] = transform(image, flip_x=j, flip_y=k, rotate=r, input_type=input_type)
                    #aug_yolo.append(transform_yolo_coords(yolo_data[i], flip_x=j, flip_y=k, rotate=r))
                    if yolo_data[i] is not None:
                        a = transform_yolo_coords(yolo_data[i], flip_x=j, flip_y=k, rotate=r)
                        aug_yolo[start + offset] = a
                    else:
                        continue
                        #print("None")
                    #print()

    return aug_images, np.asarray(aug_yolo)

def generate_labels(aug_images, num_classes):
    aug_labels = np.zeros(len(aug_images), dtype=np.uint8)
    slice = len(aug_images) // num_classes

    for i in range(0, num_classes):
        aug_labels[slice * i: slice * (i + 1)] = i

    return aug_labels


