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


def weight(shape, stddev):
    init = tf.random_normal(shape, mean=0.0, stddev=stddev)
    return tf.Variable(init)


def bias(shape, constant):
    init = tf.constant(constant, shape=shape)
    return tf.Variable(init)


def conv(input, W):
    return tf.nn.conv2d(input, W, strides=[1, 2, 2, 1], padding='SAME')


def cnn():
    ''' Convnet '''
    # TODO: build CNN architecture graph

    # layer 0 (input)
    input_layer = tf.placeholder(tf.float32, shape=[None, 1024])

    input_image = tf.reshape(input_layer, [-1, 32, 32, 1])

    # layer 1 (conv)
    W_1 = weight([5, 5, 1, 16], (0.1) / (tf.sqrt(5 * 5 * 1.)))
    b_1 = bias([16], 0.01)
    layer_1 = tf.sigmoid(conv(input_image, W_1) + b_1)

    # layer 2 (conv)
    W_2 = weight([5, 5, 16, 32], (0.1) / (tf.sqrt(5 * 5 * 16.)))
    b_2 = bias([32], 0.01)
    layer_2 = tf.sigmoid(conv(layer_1, W_2) + b_2)

    # layer 3 (conv)
    W_3 = weight([5, 5, 32, 64], (0.1) / (tf.sqrt(5 * 5 * 32.)))
    b_3 = bias([64], 0.01)
    layer_3 = tf.sigmoid(conv(layer_2, W_3) + b_3)

    # layer 4 (conv)
    W_4 = weight([4 * 4 * 64, 100], (0.1) / (tf.sqrt(1024.)))
    b_4 = bias([100], 0.1)
    layer_3_flat = tf.reshape(layer_3, [-1, 4 * 4 * 64])
    layer_4 = tf.sigmoid(tf.matmul(layer_3_flat, W_4) + b_4)

    # layer 5 (output layer)
    W_5 = weight([100, 7], (0.1) / (tf.sqrt(100.)))
    b_5 = bias([7], 0.1)
    pred_layer = tf.matmul(layer_4, W_5) + b_5

    return input_layer, pred_layer
