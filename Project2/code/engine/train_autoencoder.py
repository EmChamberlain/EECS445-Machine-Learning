"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Train Autoencoder - Skeleton
Usage: Run the command `python -m engine.train_autoencoder`
    The training progress, specifically number of batches completed and
    validation loss, will update on screen. Cause a KeyboardInterrupt
    (press CTRL-c) to end training prematurely. Whether or not training
    ends early, the model will be saved to the checkpoint specified in
    `config.json`.
"""

import tensorflow as tf

from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets
from model.build_autoencoder import autoencoder


def get_weights(saver, sess):
    ''' load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CHECKPOINT'))
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
    error = loss_func.eval(feed_dict={input_layer: faces.validation.images})
    print('\n \t mse is about %f' % error)


def train_autoencoder(input_layer, loss_func, optimizer, faces):
    ''' Train autoencoder. '''
    # TODO: ensure `config.json` specifies a number of training steps, learning
    #       rate, and batch size in accordance with the project specifications.
    #       You will not learn theory from this exercise, but you will glimpse
    #       how, sometimes, programming is plumbing: connecting up the pipes
    #       until the system works.
    try:
        for batch_index in range(get('TRAIN.NB_STEPS')):
            report_training_progress(
                batch_index, input_layer, loss_func, faces)
            batch = faces.train.next_batch(get('TRAIN.BATCH_SIZE'))
            optimizer.run(feed_dict={input_layer: batch[0]})
    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.')


if __name__ == '__main__':
    print('building model...')
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    original_image, compressed, reconstruction = autoencoder()  # fetch model layers
    rmse =  # TODO: define loss function using TensorFlow. `rmse` should depend
    #       on `original_image` and `reconstruction`.
    optimizer =  # TODO: define an optimizer per specifications. We recommend
    #       but do not require that you avoid hardcoding any
    #       constants. Instead, try calling get() to access the
    #       relevant attribute in `config.json`.
    sess.run(tf.global_variables_initializer())  # initialize some globals
    saver = tf.train.Saver()  # prepare to save model
    # load model weights if they were saved previously
    get_weights(saver, sess)

    print('loading data...')
    faces = read_data_sets()

    print('training...')
    train_autoencoder(original_image, rmse, optimizer, faces)

    print('saving trained model...\n')
    saver.save(sess, get('TRAIN.CHECKPOINT'))
