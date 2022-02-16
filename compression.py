from scipy.fftpack import fft, dct
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pywt.data
from PIL import Image
import os


if __name__ == '__main__':
    save_dir = "Compression_Plots"
    path_to_dir = os.path.join(os.getcwd(), save_dir)

    rgb_data_pickle = "rgb_data.pkl"
    rgb_data_file = open(rgb_data_pickle, 'rb')
    train_rgb = pickle.load(rgb_data_file)

    rgb_labels_pickle = "rgb_labels.pkl"
    rgb_labels_file = open(rgb_labels_pickle, 'rb')
    train_labels_rgb = pickle.load(rgb_labels_file)

    tmp = 0

    im = [train_rgb[tmp]]
    #Image.fromarray(train_rgb[tmp]).show()

    for i, label in enumerate(train_labels_rgb):
        if label != tmp:
            #print("New class at idx {0}".format(i))
            #Image.fromarray(train_rgb[i]).show()
            im.append(train_rgb[i])
            tmp = label


    for j, item in enumerate(im):
        gr_item = Image.fromarray(item).convert('L')
        gr_item = np.array(gr_item)


        num_pixels = gr_item.shape[0] * gr_item.shape[1]

        fft_transform1 = fft(gr_item.ravel()).real
        dct_transform1 = dct(gr_item.ravel(), 1)

        t = np.arange(0, num_pixels, 1)

        fig, axes = plt.subplots(1, 3)

        axes[0].set_title("Input")
        axes[1].set_title("FFT")
        axes[2].set_title("DCT")

        axes[0].imshow(item)
        axes[1].plot(t, fft_transform1)
        axes[2].plot(t, dct_transform1)

        for ax in axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fig.suptitle("Class {0}".format(j))

        #fig.show()
        plt.savefig(os.path.join(path_to_dir, str(i)))
        plt.clf()

        # Load image
        #original = pywt.data.camera()
        original = item
        # Wavelet transform of image, and plot approximation and details
        titles = ['Input', 'Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']
        coeffs2 = pywt.dwt2(original, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure(figsize=(12, 3))
        l = [original, LL, LH, HL, HH]
        for i, a in enumerate(l):
            ax = fig.add_subplot(1, len(l), i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.get_cmap('gray'))
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(path_to_dir, "Class_{0}_Wavelet_Plots".format(str(j))))