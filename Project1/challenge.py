import pandas as pd
import numpy as np
import itertools

import string as s
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import warnings
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the sentiment by df['sentiment']
    """

    return pd.read_csv(fname)


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the tweets in the heldout dataset since we will
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname + '.csv', index=False)
    return


def extract_dictionary(df):
    """
        Reads a panda dataframe, and returns a dictionary of distinct words
        mapping from the distinct word to its index (ordered by when it was found).
        Input:
          dataframe/output of load_data()
        Returns:
          a dictionary of distinct words
          mapping from the distinct word to its index (ordered by when it was found).
    """
    dict_str = ''
    for index, row in df.iterrows():
        dict_str += row['content'] + ' '

    for char in s.punctuation:
        dict_str = dict_str.replace(char, ' ')

    dict_str = dict_str.lower()
    array = np.array(dict_str.split(' '))
    unique = np.unique(array)
    if unique[0] == '':
        unique = np.delete(unique, 0)
    return unique


def generate_feature_matrix(df, word_dict):
    """
        Reads a dataframe and the dictionary of words in the reviews
        to generate {1, 0} feature vectors for each review. The resulting feature
        matrix should be of dimension (number of tweets, number of words).
        Input:
          df - dataframe that has the tweets and labels
          word_list- dictionary of words mapping to indices
        Returns:
          a feature matrix of dimension (number of tweets, number of words)
    """
    n = df.shape[0]
    d = word_dict.shape[0]
    mat = np.zeros(shape=(n, d), dtype=np.int)
    for row_index, row in df.iterrows():
        temp_str = row['content']
        for char in s.punctuation:
            temp_str = temp_str.replace(char, ' ')
            temp_str = temp_str.lower()
        temp_arr = np.array(temp_str.split(' '))
        for string in np.nditer(temp_arr):
            dict_index = np.searchsorted(word_dict, string)
            if word_dict[dict_index] == string:
                mat[row_index][dict_index] += 1

    return mat


def performance(y_true, y_pred):
    return np.float64(metrics.accuracy_score(y_true, y_pred))


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
        Splits the data, X and y, into k-folds and runs k-fold crossvalidation:
        training a classifier on K-1 folds and testing on the remaining fold.
        Calculates the k-fold crossvalidation performance metric for classifier
        clf by averaging the performance across folds.
        Input:
          clf- an instance of SVC()
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specificyin the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns: average 'test' performance across the k folds as np.float64
    """
    skf = StratifiedKFold(n_splits=k).split(X, y)
    total_performance = np.float64(0)
    count = 0
    for train, test in skf:
        count += 1
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        total_performance += performance(y_test, y_pred)
    return total_performance / count


def select_param_linear(X, y, k=5, metric="accuracy", C_range=[], penalty='l2'):
    """
        Sweeps different settings for the hyperparameter of a linear-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
          C_range - an array with all C values to be checked for
        Returns the parameter value for linear-kernel SVM, that 'maximizes' the
        average 5-fold CV performance.
    """
    best_average = 0
    best_c_val = 0
    for c in C_range:
        clf = SVC(kernel='linear', C=c, class_weight='balanced')
        current = cv_performance(clf, X, y, k=k, metric=metric)
        if current > best_average:
            best_average = current
            best_c_val = c
    return best_c_val


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specificyin the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
          parameter_values - a (num_param)x2 size numpy array which has first column as C and second column as r
          One row of parameter_values denotes one pair of values of C and r to be tried
          In grid search, it would be cartesian product of two grids
          In random search, it would be equal to same number of pairs where each one is sampled randomly
        Returns the parameter value(s) for an quadratic-kernel SVM, that 'maximize'
        the average 5-fold CV performance.
    """
    best_average = 0
    best_c_val = 0
    best_r_val = 0
    for row in param_range:
        c = row[0]
        r = row[1]
        clf = SVC(kernel='poly', degree=2, C=c, coef0=r, class_weight='balanced')
        current = cv_performance(clf, X, y, k=k, metric=metric)
        if current > best_average:
            best_average = current
            best_c_val = c
            best_r_val = r
    return best_c_val, best_r_val, best_average


def performance_CI(clf, X, y, metric="accuracy"):
    """
        Estimates the performance of clf on X,y and the corresponding 95%CI
        (lower and upper bounds)
        Input:
          clf-an instance of SVC() that has already been fit to data
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns:
            a tuple containing the performance of clf on X,y and the corresponding
            confidence interval (all three values as np.float64's)
    """
    n = X.shape[0]
    N = 100
    samples_X = np.ndarray(shape=(N, X.shape[0], X.shape[1]))
    samples_y = np.ndarray(shape=(N, y.shape[0]), dtype=np.object_)

    for sample_index in range(samples_X.shape[0]):
        for index in range(samples_X[sample_index].shape[0]):
            rand_ind = np.random.random_integers(0, n - 1)
            samples_X[sample_index][index] = X[rand_ind]
            samples_y[sample_index][index] = y[rand_ind]

    sample_performance = np.ndarray(shape=(N,))
    for index in range(samples_X.shape[0]):
        y_pred = clf.predict(samples_X[index])
        sample_performance[index] = performance(samples_y[index], y_pred)

    sample_performance = np.sort(sample_performance)
    y_pred = clf.predict(X)
    perf = performance(y, y_pred)
    lower_bound = sample_performance[np.int64(np.rint(sample_performance.shape[0] * (0.025)))]
    upper_bound = sample_performance[np.int64(np.rint(sample_performance.shape[0] * (0.975)))]

    return np.float64(perf), np.float64(lower_bound), np.float64(upper_bound)


def custom_info_gaussian_L2(X, y):
    print()
    print('custom_info_gaussian_L2**********')
    log_10_range = np.ndarray(shape=(100,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-5, 5)
    # C=np.power(10, log_10_range[index])
    best_ind = 0
    for index in range(log_10_range.shape[0]):
        mcclf = OneVsOneClassifier(
            SVC(kernel='rbf', C=np.power(10, log_10_range[index]), class_weight='balanced', random_state=0))
        perf_data[index] = cv_performance(mcclf, X, y, metric='auroc')
        if perf_data[index] > perf_data[best_ind]:
            best_ind = index

    print('Best C = ' + str(np.power(10, log_10_range[best_ind])) + ' with accuracy = ' + str(perf_data[best_ind]))
    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Gaussian L2')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()
    return np.power(10, log_10_range[best_ind])


def custom_info_linear_L2(X, y):
    print()
    print('custom_info_linear_L2**********')
    log_10_range = np.ndarray(shape=(100,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-5, 5)
    # C=np.power(10, log_10_range[index])
    best_ind = 0
    for index in range(log_10_range.shape[0]):
        mcclf = OneVsOneClassifier(
            SVC(kernel='linear', C=np.power(10, log_10_range[index]), class_weight='balanced', random_state=0))
        perf_data[index] = cv_performance(mcclf, X, y, metric='auroc')
        if perf_data[index] > perf_data[best_ind]:
            best_ind = index

    print('Best C = ' + str(np.power(10, log_10_range[best_ind])) + ' with accuracy = ' + str(perf_data[best_ind]))

    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Linear L2')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()

    return np.power(10, log_10_range[best_ind])


def custom_info_linear_L1(X, y):
    print()
    print('custom_info_linear_L1**********')
    log_10_range = np.ndarray(shape=(500,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-5, 5)
    # C=np.power(10, log_10_range[index])
    for index in range(log_10_range.shape[0]):
        mcclf = OneVsOneClassifier(
            LinearSVC(penalty='l1', dual=False, C=np.power(10, log_10_range[index]), class_weight='balanced'))
        perf_data[index] = cv_performance(mcclf, X, y, metric='auroc')

    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Linear L1')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()


def custom_info_bagging(X, y):
    print()
    print('custom_info_bagging**********')
    log_10_range = np.ndarray(shape=(50,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-3, 3)
    # C=np.power(10, log_10_range[index])
    for index in range(log_10_range.shape[0]):
        clf = SVC(kernel='linear', C=np.power(10, log_10_range[index]), class_weight='balanced')
        bagging = BaggingClassifier(base_estimator=clf, n_estimators=10, random_state=0)
        perf_data[index] = cv_performance(bagging, X, y)

    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Bagging')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()


def main():
    np.set_printoptions(edgeitems=10)
    df = load_data('challenge.csv')
    word_dict = extract_dictionary(df)
    X = generate_feature_matrix(df, word_dict)
    # y = generate_labels(df)
    y = np.array(df['sentiment'])

    df_test = load_data('held_out.csv')
    X_test = generate_feature_matrix(df_test, word_dict)

    # custom_info_gaussian_L2(X, y)
    # custom_info_linear_L2(X, y)
    # custom_info_linear_L1(X, y)
    # custom_info_bagging(X, y)

    # gaussian_L2_mcclf= OneVsRestClassifier(SVC(kernel='rbf', C=1, class_weight='balanced'))
    # linear_L2_mcclf = OneVsRestClassifier(SVC(kernel='linear', C=1, class_weight='balanced', random_state=0))
    # linear_L1_mcclf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=1, class_weight='balanced'))


    # C_val = np.power(10, -0.75)
    # linear_L1_mcclf = LinearSVC(penalty='l1', dual=False, C=C_val, class_weight='balanced')
    # linear_L1_mcclf.fit(X, y)
    # linear_perf = performance_CI(linear_L1_mcclf, X, y)
    # print('Linear | ' + str(linear_perf))
    #
    # y_pred = linear_L1_mcclf.predict(X_test)
    # generate_challenge_labels(y_pred, 'mattcham')
    #
    # y_pred_conf = linear_L1_mcclf.predict(X)
    # print(metrics.confusion_matrix(y, y_pred_conf, labels=['love', 'hate', 'sadness']))


    clf = SVC(kernel='rbf', C=1000, class_weight='balanced')
    bagging = BaggingClassifier(base_estimator=clf, n_estimators=10)
    print('Accuracy: ' + str(cv_performance(bagging, X, y)))
    bagging.fit(X, y)
    bagging_perf = performance_CI(bagging, X, y)
    print('bagging | ' + str(bagging_perf))

    y_pred_conf = bagging.predict(X)
    print(metrics.confusion_matrix(y, y_pred_conf, labels=['love', 'hate', 'sadness']))

    y_pred = bagging.predict(X_test)
    generate_challenge_labels(y_pred, 'mattcham')

    return


if __name__ == '__main__':
    main()
