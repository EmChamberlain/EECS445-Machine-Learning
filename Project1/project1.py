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
                mat[row_index][dict_index] = 1

    return mat


def performance(y_true, y_pred, y_decision, metric):
    if metric == 'accuracy':
        return np.float64(metrics.accuracy_score(y_true, y_pred))
    if metric == 'f1-score':
        # this is because a warning is thrown when F-score is 0/0 and it returns 0 in that case
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.float64(metrics.f1_score(y_true, y_pred))
    if metric == 'auroc':
        return np.float64(metrics.roc_auc_score(y_true, y_decision))
    if metric == 'precision':
        # this is because a warning is thrown when precision is 0/0 and it returns 0 in that case
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.float64(metrics.precision_score(y_true, y_pred))
    conf_mat = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tp = conf_mat[0][0]
    fn = conf_mat[0][1]
    fp = conf_mat[1][0]
    tn = conf_mat[1][1]
    if metric == 'sensitivity':
        return np.float64(tp / (tp + fn))
    if metric == 'specificity':
        return np.float64(tn / (tn + fp))


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
        y_decision = clf.decision_function(X_test)
        total_performance += performance(y_test, y_pred, y_decision, metric)
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
    N = 1000
    samples_X = np.ndarray(shape=(N, X.shape[0], X.shape[1]))
    samples_y = np.ndarray(shape=(N, y.shape[0]))

    for sample_index in range(samples_X.shape[0]):
        for index in range(samples_X[sample_index].shape[0]):
            rand_ind = np.random.random_integers(0, n - 1)
            samples_X[sample_index][index] = X[rand_ind]
            samples_y[sample_index][index] = y[rand_ind]

    sample_performance = np.ndarray(shape=(N,))
    for index in range(samples_X.shape[0]):
        y_pred = clf.predict(samples_X[index])
        y_decision = clf.decision_function(samples_X[index])
        sample_performance[index] = performance(samples_y[index], y_pred, y_decision, metric)

    sample_performance = np.sort(sample_performance)
    y_pred = clf.predict(X)
    y_decision = clf.decision_function(X)
    perf = performance(y, y_pred, y_decision, metric)
    lower_bound = sample_performance[np.int64(np.rint(sample_performance.shape[0] * (0.025)))]
    upper_bound = sample_performance[np.int64(np.rint(sample_performance.shape[0] * (0.975)))]

    return np.float64(perf), np.float64(lower_bound), np.float64(upper_bound)


def output_2_c(word_dict, X, y):
    print()
    print('2(c)**********')
    print('d = ' + str(word_dict.shape[0]))
    # np.savetxt("word_dict.csv", word_dict, fmt='%s', delimiter=',')
    # print(word_dict)
    total = 0
    count = 0
    for row in X:
        count += 1
        for col in row:
            total += col
    print('average non-zero features = ' + str(total / count))

    # for row in feature_matrix:
    #     string = ''
    #     for index in range(row.shape[0]):
    #         if row[index] == 1:
    #             string+= word_dict[index] + ' '
    #     print(string)
    # np.savetxt("feature_matrix.csv", feature_matrix, fmt='%s', delimiter=',')
    # print(feature_matrix)
    print('DONE**********')
    print()


def output_3_1_c(X, y):
    print()
    print('3.1(c)**********')
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    Metrics = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    for metric in Metrics:
        c_val = select_param_linear(X, y, metric=metric, C_range=C_range)
        clf = SVC(kernel='linear', C=c_val, class_weight='balanced')
        performance = cv_performance(clf, X, y, metric=metric)
        print(metric + ' | ' + str(c_val) + ' | ' + str(performance))
    print('DONE**********')
    print()


def output_3_1_d(X, y):
    print()
    print('3.1(d)*******')
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    L0_data = np.zeros(shape=(7,))
    exp = np.array([-3, -2, -1, 0, 1, 2, 3])
    index = 0
    for c in C_range:
        clf = SVC(kernel='linear', C=c, class_weight='balanced')
        clf.fit(X, y)
        theta_vec = clf.coef_[0]
        L0_norm = 0
        for theta in theta_vec:
            if not theta == 0:
                L0_norm += 1
        L0_data[index] = L0_norm
        index += 1

    plt.figure()
    plt.plot(exp, L0_data, '-or', label='L0_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('L0_norm', fontsize=16)
    plt.title('Part 3.1(d)')
    plt.show()
    print('DONE**********')
    print()


def output_3_2_b(X, y):
    print()
    print('3.2(b)*******')
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    R_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    grid_search = np.ndarray(shape=(49, 2))
    index = 0
    for c in C_range:
        for r in R_range:
            grid_search[index][0] = c
            grid_search[index][1] = r
            index += 1

    c_grid, r_grid, perf_grid = select_param_quadratic(X, y, metric='auroc', param_range=grid_search)
    print('Grid Search | ' + str(c_grid) + ' | ' + str(r_grid) + ' | ' + str(perf_grid))

    random_search = np.ndarray(shape=(25, 2))
    for row in random_search:
        row[0] = np.power(10, np.random.uniform(-3, 3))
        row[1] = np.power(10, np.random.uniform(-3, 3))

    c_random, r_random, perf_random = select_param_quadratic(X, y, metric='auroc', param_range=random_search)
    print('Random Search | ' + str(c_random) + ' | ' + str(r_random) + ' | ' + str(perf_random))

    print('DONE**********')
    print()


def output_3_4_a(X, y):
    print()
    print('3.4(a)*******')
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    best_average = 0
    best_c_val = 0
    for c in C_range:
        clf = LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced')
        current = cv_performance(clf, X, y, metric='auroc')
        if current > best_average:
            best_average = current
            best_c_val = c

    print('Grid Search | ' + str(best_c_val) + ' | ' + str(best_average))

    print('DONE**********')
    print()


def output_3_4_b(X, y):
    print()
    print('3.4(b)*******')
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    L0_data = np.zeros(shape=(7,))
    exp = np.array([-3, -2, -1, 0, 1, 2, 3])
    index = 0
    for c in C_range:
        clf = LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced')
        clf.fit(X, y)
        theta_vec = clf.coef_[0]
        L0_norm = 0
        for theta in theta_vec:
            if not theta == 0:
                L0_norm += 1
        L0_data[index] = L0_norm
        index += 1

    plt.figure()
    plt.plot(exp, L0_data, '-or', label='L0_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('L0_norm', fontsize=16)
    plt.title('Part 3.4(b)')
    plt.show()
    print('DONE**********')
    print()


def custom_info_linear_L2(X, y):
    print()
    print('custom_info_linear_L2**********')
    log_10_range = np.ndarray(shape=(100,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-5, 5)

    for index in range(log_10_range.shape[0]):
        clf = SVC(kernel='linear', C=np.power(10, log_10_range[index]), class_weight='balanced')
        perf_data[index] = cv_performance(clf, X, y, metric='auroc')

        theta_vec = clf.coef_[0]
        L0_norm = 0
        for theta in theta_vec:
            if not theta == 0:
                L0_norm += 1
        L0_norm_data[index] = L0_norm

    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('AUROC', fontsize=16)
    plt.title('Performance vs C')

    plt.figure()
    plt.plot(log_10_range, L0_norm_data, 'or', label='L0_norm_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('L0_norm', fontsize=16)
    plt.title('L0_norm vs C')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()


def custom_info_linear_L1(X, y):
    print()
    print('custom_info_linear_L1**********')
    log_10_range = np.ndarray(shape=(500,))
    perf_data = np.ndarray(shape=log_10_range.shape)
    L0_norm_data = np.ndarray(shape=log_10_range.shape)
    for i in range(log_10_range.shape[0]):
        log_10_range[i] = np.random.uniform(-5, 5)

    for index in range(log_10_range.shape[0]):
        clf = LinearSVC(penalty='l1', dual=False, C=np.power(10, log_10_range[index]), class_weight='balanced')
        perf_data[index] = cv_performance(clf, X, y, metric='auroc')

        theta_vec = clf.coef_[0]
        L0_norm = 0
        for theta in theta_vec:
            if not theta == 0:
                L0_norm += 1
        L0_norm_data[index] = L0_norm

    plt.figure()
    plt.plot(log_10_range, perf_data, 'or', label='perf_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('AUROC', fontsize=16)
    plt.title('Performance vs C')

    plt.figure()
    plt.plot(log_10_range, L0_norm_data, 'or', label='L0_norm_data')
    plt.xlabel('C = 10^x', fontsize=16)
    plt.ylabel('L0_norm', fontsize=16)
    plt.title('L0_norm vs C')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()


def custom_info_quadratic_L2(X, y):
    print()
    print('custom_info_quadratic_L2**********')
    log_10_range_C = np.ndarray(shape=(100,))
    log_10_range_R = np.ndarray(shape=log_10_range_C.shape)
    perf_data = np.ndarray(shape=log_10_range_C.shape)
    for i in range(log_10_range_C.shape[0]):
        log_10_range_C[i] = np.random.uniform(-5, 5)
        log_10_range_R[i] = np.random.uniform(-5, 5)

    for index in range(log_10_range_C.shape[0]):
        clf = SVC(kernel='poly', degree=2, C=np.power(10, log_10_range_C[index]),
                  coef0=np.power(10, log_10_range_R[index]), class_weight='balanced')
        perf_data[index] = cv_performance(clf, X, y, metric='auroc')
        index += 1

    figure_perf = plt.figure()
    ax_perf = figure_perf.add_subplot(111, projection='3d')
    ax_perf.scatter(log_10_range_C, log_10_range_R, perf_data, 'or', label='perf_data')
    ax_perf.set_xlabel('C')
    ax_perf.set_ylabel('r')
    ax_perf.set_zlabel('AUROC')
    plt.title('Performance vs C vs r')

    print('Showing plot...')
    plt.show()
    print('DONE**********')
    print()


def output_4_b(linear_L2_clf, linear_L1_clf, quadratic_L2_clf, X_test, y_test):
    print()
    print('4(b)*******')
    linear_L2_performance, linear_L2_lower_bound, linear_L2_upper_bound \
        = performance_CI(linear_L2_clf, X_test, y_test, 'auroc')
    print('Linear-kernel SVM with hinge loss and L2-penalty | ' + str(linear_L2_performance) + ' | ('
          + str(linear_L2_lower_bound) + ', ' + str(linear_L2_upper_bound) + ')')

    linear_L1_performance, linear_L1_lower_bound, linear_L1_upper_bound \
        = performance_CI(linear_L1_clf, X_test, y_test, 'auroc')
    print('Linear-kernel SVM with squared hinge loss and L1-penalty | ' + str(linear_L1_performance) + ' | ('
          + str(linear_L1_lower_bound) + ', ' + str(linear_L1_upper_bound) + ')')

    quadratic_L2_performance, quadratic_L2_lower_bound, quadratic_L2_upper_bound \
        = performance_CI(quadratic_L2_clf, X_test, y_test, 'auroc')
    print('Quadratic-kernel SVM hinge loss and L2-penalty | ' + str(quadratic_L2_performance) + ' | ('
          + str(quadratic_L2_lower_bound) + ', ' + str(quadratic_L2_upper_bound) + ')')

    print('DONE**********')
    print()


def main():
    np.set_printoptions(edgeitems=10)
    df = load_data('dataset.csv')
    word_dict = extract_dictionary(df)
    X = generate_feature_matrix(df, word_dict)
    y = np.array(df['label'])
    X_train = X[:400]
    y_train = y[:400]
    X_test = X[400:]
    y_test = y[400:]

    # output_2_c(word_dict, X, y)
    # output_3_1_c(X_train, y_train)
    # output_3_1_d(X_train, y_train)
    # output_3_2_b(X_train, y_train)
    # output_3_4_a(X_train, y_train)
    # output_3_4_b(X_train, y_train)

    # custom_info_linear_L2(X_train, y_train)
    # custom_info_linear_L1(X_train, y_train)
    # custom_info_quadratic_L2(X_train, y_train)

    linear_L2_clf = SVC(kernel='linear', C=1, class_weight='balanced')
    linear_L2_clf.fit(X_train, y_train)

    linear_L1_clf = LinearSVC(penalty='l1', dual=False, C=1e-1, class_weight='balanced')
    linear_L1_clf.fit(X_train, y_train)

    quadratic_L2_clf = SVC(kernel='poly', degree=2, C=1e2, coef0=1e2, class_weight='balanced')
    quadratic_L2_clf.fit(X_train, y_train)

    # output_4_b(linear_L2_clf, linear_L1_clf, quadratic_L2_clf, X_test, y_test)

    # thetas = linear_L1_clf.coef_[0]
    #
    # for i in range(thetas.shape[0]):
    #     if not thetas[i] == 0:
    #         print(str(thetas[i]) + ': ' + word_dict[i])
    return


if __name__ == '__main__':
    main()
