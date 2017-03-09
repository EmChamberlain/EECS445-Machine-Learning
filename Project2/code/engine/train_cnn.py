"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Train CNN - Skeleton
Usage: Run the command `python -m engine.train_cnn`
    The training progress, specifically number of batches completed and
    validation loss, will update on screen. Cause a KeyboardInterrupt
    (press CTRL-c) to end training prematurely. Whether or not training
    ends early, the model will be saved to the checkpoint specified in
    `config.json`.
"""

import tensorflow as tf
from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets
from model.build_cnn import cnn


def get_weights(saver, sess):
    ''' load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
        print('Yay! I restored weights from a saved model!')
    else:
        print('OK, I did not find a saved model, so I will start training from scratch!')


def report_training_progress(batch_index, input_layer, loss_func, faces):
    ''' Update user on training progress '''
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return
    error = loss_func.eval(feed_dict={input_layer: faces.validation.images[
                           :2000], true_labels: faces.validation.labels[:2000]})
    acc = accuracy.eval(feed_dict={input_layer: faces.validation.images[
                        :2000], true_labels: faces.validation.labels[:2000]})
    # error = loss_func.eval(feed_dict={input_layer: faces.test.images, true_labels: faces.test.labels})
    # acc = accuracy.eval(feed_dict={input_layer: faces.test.images, true_labels: faces.test.labels})
    print('\n \t cross_entropy is about %f' % error)
    print(' \t accuracy is about %f' % acc)


def train_cnn(input_layer, prediction_layer, loss_func, optimizer, faces):
    ''' Train CNN '''
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(
                batch_index, input_layer, loss_func, faces)
            batch_images, batch_labels = faces.train.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_labels: batch_labels})
    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.')


if __name__ == '__main__':
    print('building model...')
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    input_layer, prediction_layer = cnn()  # fetch model layers
    true_labels = tf.placeholder(tf.float32, shape=[None, 7])
    cross_entropy =  # TODO: define the loss function
    correct_prediction =  # TODO: define the correct predictions calculation
    accuracy =  # TODO: calculate accuracy
    optimizer =  # TODO: define the training step
    sess.run(tf.global_variables_initializer())  # initialize some globals
    saver = tf.train.Saver()  # prepare to save model
    # load model weights if they were saved previously
    get_weights(saver, sess)

    print('loading data...')
    faces = read_data_sets(one_hot=True)

    print('training...')
    train_cnn(input_layer, prediction_layer, cross_entropy, optimizer, faces)

    print('saving trained model...\n')
    saver.save(sess, get('TRAIN.CNN.CHECKPOINT'))
