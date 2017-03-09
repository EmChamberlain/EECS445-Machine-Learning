"""
EECS 445 - Winter 2017 - Project 2
FER2013 - Skeleton
This file reads the dataset and provides a function `preprocessed_data`
that returns preprocessed images, labels
Usage: python -m data_scripts.fer2013
"""

import numpy as np
from scipy.misc import imresize
from sklearn.utils import resample
import pandas

from utils.config import get, print_if_verbose


class FER2013:
    filename = ''
    data_stored = False
    train_images = np.zeros(0)
    train_labels = np.zeros(0)
    val_images = np.zeros(0)
    val_labels = np.zeros(0)
    test_images = np.zeros(0)
    test_labels = np.zeros(0)

    def __init__(self):
        self.filename = get('DATA.FER_PATH')
        self.data_stored = False

    def get_images_labels(self, matrix):
        image_row_len = len(np.fromstring(matrix[0, 1], dtype=int, sep=' '))
        image_dim = int(np.sqrt(image_row_len))
        # images = np.zeros((matrix.shape[0], image_dim, image_dim))
        labels = matrix[:, 0]
        images = []
        for i in range(matrix.shape[0]):
            image_row = np.fromstring(matrix[i, 1], dtype=int, sep=' ')
            images.append(np.reshape(image_row, (image_dim, image_dim)))
        images = np.array(images)
        return images, labels

    def read_csv(self):
        df = pandas.read_csv(self.filename)
        mat = df.as_matrix()
        train_mat = mat[mat[:, 2] == 'Training', :]
        val_mat = mat[mat[:, 2] == 'PublicTest', :]
        test_mat = mat[mat[:, 2] == 'PrivateTest', :]
        self.train_images, self.train_labels = self.get_images_labels(
            train_mat)
        self.val_images, self.val_labels = self.get_images_labels(val_mat)
        self.test_images, self.test_labels = self.get_images_labels(test_mat)
        self.data_stored = True

    def balance_classes(self, images, labels, count=5000):
        balanced_images, balanced_labels = [], []
        unique_labels = set(labels)
        for l in unique_labels:
            l_idx = np.where(labels == l)[0]
            l_images, l_labels = images[l_idx], labels[l_idx]
            # Consistent resampling to facilitate debugging
            resampled_images, resampled_labels = resample(l_images,
                                                          l_labels,
                                                          n_samples=count,
                                                          random_state=0)
            balanced_images.extend(resampled_images)
            balanced_labels.extend(resampled_labels)
        balanced_images = np.array(balanced_images)
        balanced_labels = np.array(balanced_labels)
        print('---Shuffled images shape: {}'.format(balanced_images.shape))
        print('---Shuffled labels shape: {}'.format(balanced_labels.shape))
        assert(len(balanced_images) == len(balanced_labels))
        shuffle_idx = np.random.permutation(len(balanced_images))
        return balanced_images[shuffle_idx], balanced_labels[shuffle_idx]

    def resize(self, images, new_size=32):
        resized = []
        for i in range(images.shape[0]):
            resized_image = imresize(images[i],
                                     size=(new_size, new_size),
                                     interp='bicubic')
            resized.append(resized_image)
        return np.array(resized)

    def preprocessed_data(self, split, dim=32, one_hot=False, balance_classes=True):
        if not self.data_stored:
            self.read_csv()
        if split == 'train':
            print_if_verbose('Loading train data...')
            images, labels = self.train_images, self.train_labels
            # TODO: Remove blank images
            if balance_classes:
                images, labels = self.balance_classes(images, labels, 5000)
        elif split == 'val':
            print_if_verbose('Loading validation data...')
            images, labels = self.val_images, self.val_labels
            # TODO: Remove blank images
            if balance_classes:
                images, labels = self.balance_classes(images, labels, 500)
        elif split == 'test':
            print_if_verbose('Loading test data...')
            images, labels = self.test_images, self.test_labels
            # TODO: Remove blank images
            if balance_classes:
                images, labels = self.balance_classes(images, labels, 500)
        else:
            print_if_verbose('Invalid input!')
            return
        images = self.resize(images, dim)
        # TODO: Normalize, add dimension, one-hot encoding of labels

        print_if_verbose('---Images shape: {}'.format(images.shape))
        print_if_verbose('---Labels shape: {}'.format(labels.shape))
        return images, labels


if __name__ == '__main__':
    data = FER2013()
    images, labels = data.preprocessed_data('train')  # 'train' or 'val' or 'test'
