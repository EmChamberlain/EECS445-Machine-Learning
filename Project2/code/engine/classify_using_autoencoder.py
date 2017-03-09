"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Use autoencoder representation as feature vector for image classifier
Usage: Run the command `python -m engine.classify_using_autoencoder`
       to view classification score
"""

import tensorflow as tf
from model.build_autoencoder import autoencoder
from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets

from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    assert is_file_prefix(
        'TRAIN.AUTOENCODER.CHECKPOINT'), 'training checkpoint not found!'
    print('building model...')
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    original_image, compressed, reconstruction = autoencoder()  # fetch model layers
    sess.run(tf.global_variables_initializer())  # initialize some globals
    saver = tf.train.Saver()  # prepare to restore model
    saver.restore(sess, get('TRAIN.AUTOENCODER.CHECKPOINT'))
    print('Yay! I restored weights from a saved model!')

    print('loading data...')
    faces = read_data_sets(one_hot=False)

    print('training classifier...')
    compressed_trainset = compressed.eval(
        feed_dict={original_image: faces.train.images})
    clf = LogisticRegression()  # TODO (optional): adjust hyperparameters
    clf.fit(compressed_trainset, faces.train.labels)

    print('testing classfier...')
    compressed_testset = compressed.eval(
        feed_dict={original_image: faces.test.images})
    accuracy = clf.score(compressed_testset, faces.test.labels)
    print('Autoencoder-based classifier achieves accuracy \n%f' % accuracy)
