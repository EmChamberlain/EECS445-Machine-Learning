import pandas as pd
import numpy as np
import itertools

import string as s
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt;
from sklearn.metrics import confusion_matrix


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
        return metrics.accuracy_score(y_true, y_pred)
    if metric == 'f1-score':
        return metrics.f1_score(y_true, y_pred)
    if metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_decision)
    if metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tp = conf_mat[0][0]
    fn = conf_mat[0][1]
    fp = conf_mat[1][0]
    tn = conf_mat[1][1]
    if metric == 'sensitivity':
        return tp / (tp + fn)
    if metric == 'specificity':
        return tn / (tn + fp)
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
    n = X.shape[0]
    d = X.shape[1]
    skf = StratifiedKFold.StratifiedKFold(y, k=k)
    total_performance = np.float64
    count = 0
    for train, test in skf:
        count+=1
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_decision = clf.decision_function(X_test)
        total_performance+=performance(y_test, y_pred, y_decision, metric)
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


def performance(y_true, y_pred, metric="accuracy"):
    """
          Calculates the performance metric based on the agreement between the
          true labels and the predicted labels
          Input:
            y_true- (n,) array containing known labels
            y_pred- (n,) array containing predicted scores
            metric- string option used to select the performance measure
          Returns: the performance as a np.float64
    """


def main():
    np.set_printoptions(edgeitems=10)
    df = load_data('dataset.csv')
    word_dict = extract_dictionary(df)
    print('2.c**********')
    print('d = ' + str(word_dict.shape[0]))
    # np.savetxt("word_dict.csv", word_dict, fmt='%s', delimiter=',')
    # print(word_dict)
    feature_matrix = generate_feature_matrix(df, word_dict)
    total = 0
    count = 0
    for row in feature_matrix:
        count+=1
        for col in row:
            total += col
    print('average non-zero features = ' + str(total/count))

    # for row in feature_matrix:
    #     string = ''
    #     for index in range(row.shape[0]):
    #         if row[index] == 1:
    #             string+= word_dict[index] + ' '
    #     print(string)
    # np.savetxt("feature_matrix.csv", feature_matrix, fmt='%s', delimiter=',')
    # print(feature_matrix)
    return


if __name__ == '__main__':
    main()
