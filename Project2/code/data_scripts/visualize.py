"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Sample and visualize face images
Usage: python -m data_scripts.visualize

This will open up a window displaying a randomly selected validation
image. The label of the image is shown. When the window is closed,
another window will pop up with a different image, and so on. Exit
this loop by causing a KeyboardInterrupt (press CTRL-c) in the
terminal.

If the face data has not been used before, then this script will
spend some time unpacking the .csv, doing pre-processing, and saving
the data into .npy array for faster future loading. The success of
this script is a good indication that the data flow part of this
project (something the IAs have figured out and that hopefully will
not bother the students) is running smoothly. If this script fails,
please ask course staff for assistance.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_scripts.fer2013_dataset import read_data_sets
from collections import Counter

print('reading data...')
faces = read_data_sets(one_hot=False)
Xs = faces.validation.images
ys = faces.validation.labels
print(ys)
print('Label distribution in valset is:', Counter(ys))

print('I will display some images. Press CTRL-c to exit.')
while True:
    index = np.random.choice(len(Xs))
    plt.imshow(Xs[index].reshape((32, 32)), plt.get_cmap(
        'gray'), interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title('Label %d' % ys[index])
    plt.show()
