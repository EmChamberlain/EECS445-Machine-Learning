"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Visually compare autoencoder to naive compression scheme.
Usage: Run the command `python -m engine.visualize_autoencoder`
    Then enter in labels in [0, 7) into the prompt to visualize
    autoencoder behavior on a randomly selected image of a corresponding
    class. Specifically, shown side-by-side will be the original image,
    a naive reconstruction obtained by downsampling-then-upsampling, and
    the autoencoder reconstruction. Exit by causing a KeyboardInterrupt
    (press CTRL-c).
"""

import numpy as np
import tensorflow as tf
from model.build_autoencoder import naive, autoencoder
from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets
import matplotlib.pyplot as plt


def plot(subplot_index, image, name, nb_subplots=3):
    ''' Plot a given image side-by-side the previously plotted ones. '''
    plt.subplot(1, nb_subplots, subplot_index + 1)
    plt.imshow(image, plt.get_cmap('gray'),
               interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':
    print('restoring model...')
    assert is_file_prefix(
        'TRAIN.AUTOENCODER.CHECKPOINT'), "training checkpoint not found!"
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    auto_orig, auto_repr, auto_recon = autoencoder()  # fetch autoencoder layers
    naive_orig, naive_repr, naive_recon = naive()  # fetch naive baseline layers
    saver = tf.train.Saver()  # prepare to restore weights
    saver.restore(sess, get('TRAIN.AUTOENCODER.CHECKPOINT'))
    print('Yay! I restored weights from a saved model!')

    print('loading data...')
    faces = read_data_sets()
    ys = faces.validation.labels
    Xs = faces.validation.images
    Xs_baseline = np.ones(Xs.shape)

    print('computing reconstructions...')
    As = auto_recon.eval(feed_dict={auto_orig: Xs})

    rmse = np.sqrt(np.mean(np.square(np.subtract(Xs, As)), axis=1))
    rmse_baseline = np.sqrt(np.mean(np.square(np.subtract(Xs, Xs_baseline)), axis=1))

    classes = [[], [], [], [], [], [], []]
    for i in range(ys.shape[0]):
        for j in range(ys[i].shape[0]):
            if ys[i][j] == 1:
                classes[j].append(i)

    print('Baseline RMSE: ' + str(np.mean(rmse_baseline)))
    curr_class = 0
    for cl in classes:
        argmin = cl[0]
        argmax = cl[0]
        argavg = cl[0]

        total_rmse = 0
        num = 0

        for i in cl:
            if rmse[i] < rmse[argmin]:
                argmin = i
            if rmse[i] > rmse[argmax]:
                argmax = i
            rmseavg = (rmse[argmin] + rmse[argmax]) / 2
            if np.abs(rmseavg - rmse[i]) < np.abs(rmseavg - rmse[argavg]):
                argavg = i
            total_rmse += rmse[i]
            num += 1
        print('Class ' + str(curr_class) + ' average error: ' + str(total_rmse / num))
        curr_class += 1

        plot(0, Xs[argmin].reshape(32, 32), 'best case')
        plot(1, Xs[argmax].reshape(32, 32), 'worst case')
        plot(2, Xs[argavg].reshape(32, 32), 'typical case')
        plt.show()
