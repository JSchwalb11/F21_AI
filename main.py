import pickle
import numpy as np
import data_augmentation
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from sklearn.datasets import make_circles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from time import time as now
import argparse
import keras



def slice(images):
    test_ratio = 0.05

    slice1 = int(len(images) * test_ratio)
    slice2 = int(len(images) / 2)
    slice3 = int(len(images) / 2 + len(images) * test_ratio)

    test_images = images[:slice1]
    train_images = images[slice1:slice2]
    test_images.extend(images[slice2:slice3])
    train_images.extend(images[slice3:])

    return train_images, test_images

def get_labels(train_images, test_images):
    train_labels = np.zeros(len(train_images))
    test_labels = np.zeros(len(test_images))

    # First half of each set has 0 hostages, Second half has 1 hostage
    # Training Set
    for i in range(0, len(train_images)):
        if i >= len(train_images) / 2:
            train_labels[i] += 1
    # Testing Set
    for i in range(0, len(test_images)):
        if i >= len(test_images):
            test_labels[i] += 1

    return train_labels.astype(np.int64), test_labels.astype(np.int64)

def normalize(train_images, test_images):
    train_images = (np.expand_dims(train_images, axis=-1) / 255.).astype(np.float32)
    test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)

    return train_images, test_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug',  type=bool,
                        help='Keep true unless working on a machine with large memory (75gb+)')
    args = parser.parse_args()

    train_file = open("train_data", 'rb')
    test_file = open("test_data", 'rb')

    images = pickle.load(test_file)
    images = np.asarray(images)
    print("Images.shape: {0}".format(images.shape))

    test_img = images[0]
    """
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
    rotate = [0, 90, 180, 270]
    #rotate = [0, 90]

    permutations = len(flip_x) * len(flip_y) * len(rotate)

    aug_images = np.zeros((images.shape[0] * permutations, images.shape[1], images.shape[2], images.shape[3]), dtype=np.uint8)
    aug_labels = np.zeros((images.shape[0] * permutations), dtype=np.uint8)

    for i, image in enumerate(images):
        for i_j, j in enumerate(flip_x):
            for i_k, k in enumerate(flip_y):
                for i_r, r in enumerate(rotate):
                    offset = i_j + i_k + i_r
                    start = i * permutations
                    end = start + offset
                    print("start + offset = {0} + {1}".format(start, offset))
                    print("aug_images.shape[0] = {0}".format(aug_images.shape[0]))
                    aug_images[start + offset] = data_augmentation.transform(image, flip_x=j, flip_y=k, rotate=r)
                    #print("Transform params:\nflip_x {0}\n flip_y {1}\n rotation {2}\n".format(j,k,r))

    aug_labels[len(aug_images)//2:] = 1

    # create model

    if args.debug == True:
        dim = 256
    else:
        dim = 512

    train_images, test_images, train_labels, test_labels = train_test_split( \
        aug_images[:, :dim, :dim], aug_labels, test_size=0.15, random_state=22)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    """
    for i in range(0, EPOCHS):
        print("Starting epoch {0}".format(i))
        print("Shapes...\ntrain_labels {0}\n train_images {1}\n".format(train_labels.shape, train_images.shape))
        p = np.random.permutation(len(train_images))
        model.fit(train_images[p], train_labels, batch_size=BATCH_SIZE, epochs=1)
        f1_score = f1_score(test_labels, model.predict(test_images))
    """
    X, y = train_images, train_labels

    model = keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(dim,dim,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    """
    # define model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (dim, dim, 3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    """
    # fit model

    """
    
    for i in range(0, EPOCHS):
        p = np.random.permutation(len(train_images))
        lap = now() - start
        history = model.fit(X[p], y, validation_data=(test_images, test_labels), batch_size=BATCH_SIZE, epochs=1)
        print("completed training in %.3f" % (now()-lap))
    """
    BATCH_SIZE = 20
    EPOCHS = 500

    p = np.random.permutation(len(train_images))
    start = now()
    history = model.fit(X, y, validation_data=(test_images, test_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)
    print("Total training time in %.3f" % (now() - start))

    # evaluate the model
    _, train_acc = model.evaluate(train_images, train_labels, verbose=0)
    _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


