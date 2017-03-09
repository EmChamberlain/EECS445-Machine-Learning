"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Clustering Utils
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if os.path.exists(os.path.realpath('data_scripts/fer2013_solution.py')):
    from data_scripts.fer2013_solution import FER2013
else:
    from data_scripts.fer2013 import FER2013

image_size = (32, 32)


def get_data():
    """
    Input:
        Imports the FER2013 Faces Data
    Returns:
        X: an n x d array, in which each row represents an image
        y: a n X 1 vector, elements of which are integers between 0 and nc-1
           where nc is the number of classes represented in the data
    """
    data = FER2013()
    images, labels = data.preprocessed_data('train', one_hot=False)  # 'train' or 'val'
    images = images[:500]
    X = np.zeros((len(images), 1024))
    for i in range(len(images)):
        X[i, :] = np.asarray(images[i]).flatten()
    return X, labels


def show_im(im, size=image_size):
    """
    Input:
        im: a row or column vector of dimension d
        size: a pair of positive integers (i, j) such that i * j = d
              defaults to the right value for our images
    Opens a new window and displays the image
    """
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap=cm.gray)
    plt.show()
