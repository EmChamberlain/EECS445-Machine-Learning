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


def get_index_from_user_supplied_label(ys):
    ''' Return index (into validation set) corresponding to user-supplied
        label.
    '''
    while True:
        try:
            label = int(input('Enter label in [0, 7): '))
            assert(0 <= label < 7)
            break
        except ValueError:
            print('Oops! I need an integer...')
        except AssertionError:
            print('Oops! Valid labels are in [0, 7)...')
    while True:
        index = np.random.choice(len(ys))
        if ys[index] == label:
            return index


def plot(subplot_index, image, name, nb_subplots=3):
    ''' Plot a given image side-by-side the previously plotted ones. '''
    plt.subplot(1, nb_subplots, subplot_index + 1)
    plt.imshow(image, plt.get_cmap('gray'),
               interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


def user_interaction_loop(ys, orig_images, naive_recons, auto_recons):
    ''' Main loop: user enters labels to produce plots '''
    sl = get('MODEL.SQRT_REPR_DIM')
    try:
        while True:
            index = get_index_from_user_supplied_label(ys)
            plot(0, orig_images[index].reshape(32, 32), 'original image')
            plot(1, naive_recons[index].reshape(sl, sl), 'naive recon')
            plot(2, auto_recons[index].reshape(32, 32), 'autoencoder recon')
            plt.show()
    except KeyboardInterrupt:
        print('OK, bye!')

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

    print('computing reconstructions...')
    Ns = naive_recon.eval(feed_dict={naive_orig: Xs})
    As = auto_recon.eval(feed_dict={auto_orig: Xs})

    print('starting visualization...')
    user_interaction_loop(ys, Xs, Ns, As)
