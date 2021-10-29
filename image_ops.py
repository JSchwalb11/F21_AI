import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def get_contours(im, input_type):
    if input_type != 'b/w':
        bw_img = get_bw_image(im)
    else:
        bw_img = im
    contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    #for cnt in contours:
        # += 1
        #x, y, w, h = cv2.boundingRect(cnt)
        #roi = im[y:y + h, x:x + w]
        #cv2.imwrite(str(idx) + '.jpg', roi)
        #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
    #cv2.imshow('img', im)
    #cv2.waitKey(0)
    return im, contours

def get_bw_image(im_rgb):
    im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return bw_img

def get_canny_edge(im_rgb, threshold1=100, threshold2=200):
    canny = cv2.Canny(im_rgb, threshold1, threshold2)
    return canny

def extract_edges(numpy_img_arr):
    edge_arr = np.zeros((numpy_img_arr.shape[0], numpy_img_arr.shape[1], numpy_img_arr.shape[2]),
                          dtype=np.uint8)
    for i, image in enumerate(numpy_img_arr):
        edge_arr[i] = get_canny_edge(image)

    return edge_arr

def filter_image_contours(idx, single_image, canny_edge=False, contours=False):
    im = get_canny_edge(single_image)
    im2, contours = get_contours(im, 'b/w')
    max_area = 0
    max_area_arg = 0
    area = lambda cnt: math.prod(cv2.boundingRect(cnt)[2:])


    for i, cnt in enumerate(contours):
        #area_i = cv2.contourArea(cnt)#area(cnt)
        area_i = area(cnt)
        if area_i > max_area:
            max_area = area_i
            max_area_arg = i
    data = []
    if len(contours) > 0:
        object_of_interest = contours[max_area_arg]
        x, y, w, h = cv2.boundingRect(object_of_interest)
        im3 = np.zeros((im2.shape[0], im2.shape[1]), dtype=np.uint8)
        if max_area > 512 and w < 128:
            object_of_interest = contours[max_area_arg]
            x, y, w, h = cv2.boundingRect(object_of_interest)
            # print("x: {0}\ny: {1}\nw: {2}\nh: {3}\n".format(x,y,w,h))
            # cv2.rectangle(im3, (x, y), (x + w, y + h), (200, 0, 0), 2)
            im3[y:y + h, x:x + w] = im2[y:y + h, x:x + w]
            #data = (x, y, w, h)
            return im3, np.asarray([x, y, w, h])
        else:
            #fig, axs = plt.subplots(2)
            #cv2.rectangle(im3, (x, y), (x + w, y + h), (200, 0, 0), 2)
            #axs[0].imshow(im2)
            #axs[1].imshow(im3)
            #plt.show()
            #print("Found something weird")
            #print("x: {0}\ny: {1}\nw: {2}\nh: {3}\n".format(x,y,w,h))
            return None, None


    else:
        print("Flag image {0} for review".format(idx))
        return 1, None, None


def filter_image_array_contours(image_array):
    tmp = np.zeros((image_array.shape[0], image_array.shape[1], image_array.shape[2]))
    img_coords = list()
    for i, single_image in enumerate(image_array):
        print("image #{0}".format(i))
        img, data = filter_image_contours(i, single_image)
        tmp[i] = img
        img_coords.append(data)
    return tmp, np.asarray(img_coords)

if __name__ == '__main__':
    """
    import pickle
    import sys
    from matplotlib import pyplot as plt


    aug_images_file = open("aug_data.pkl", 'rb')
    aug_labels_file = open("aug_labels.pkl", 'rb')

    aug_images = pickle.load(aug_images_file)
    aug_labels = pickle.load(aug_labels_file)

    if (len(aug_images) == len(aug_labels)):
        print("Lengths match")
    else:
        print("Lengths do not match, check data loader")
        sys.exit(0)

    num_classes = 4
    slice = len(aug_images) // num_classes
    im = aug_images[slice*1+1000]
    im_bw = get_bw_image(im)
    im_contour = get_contours(im_bw, input_type='b/w')
    edges = cv2.Canny(im, 100, 200)
    bw_edges = cv2.Canny(im_bw, 100, 200)
    plt.figure(1)

    plt.subplot(211)
    plt.imshow(edges, cmap='gray')
    plt.subplot(212)
    plt.imshow(bw_edges, cmap='gray')
    #plt.imshow(im_contour, cmap='gray')
    #plt.subplot(221)
    #plt.imshow(edges, cmap='gray')
    plt.show()
    """