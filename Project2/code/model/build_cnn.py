"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Build CNN - Skeleton
Build TensorFlow computation graph for convolutional network
Usage: `from model.build_cnn import cnn`
"""

import tensorflow as tf


# TODO: can define helper functions here to build CNN graph

def normalize(x):
    ''' Set mean to 0.0 and standard deviation to 1.0 via affine transform '''
    shifted = x - tf.reduce_mean(x)
    scaled = shifted / tf.sqrt(tf.reduce_mean(tf.multiply(shifted, shifted)))
    return scaled


def cnn():
    ''' Convnet '''
    # TODO: build CNN architecture graph
    return input_layer, pred_layer
