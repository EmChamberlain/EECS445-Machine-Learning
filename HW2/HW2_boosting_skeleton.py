"""
EECS 445 - Introduction to Machine Learning
Winter 2017
Homework 2, AdaBoost
Skeleton Code
"""

import numpy as np
import matplotlib.pyplot as plt

# LABELLED DATA
X = np.array([[ 0.91281307, 0.10953233],\
              [ 0.08718399, 0.82802638],\
              [ 0.23744548, 0.08097967],\
              [ 0.47536559, 0.30639172],\
              [ 0.67336331, 0.25726507],\
              [ 0.84573874, 0.26380655],\
              [ 0.44222202, 0.14736479],\
              [ 0.46680658, 0.54903108],\
              [ 0.22367281, 0.63011133],\
              [ 0.87236129, 0.4586615 ]])

labels = np.array([1, -1, -1, 1, 1, 1, 1, -1, -1, -1])
# END OF DATA

# HELPER FUNCTIONS
def stumpClassificationResult(feature, threshold, compare, X):
    '''
    Input: decision stump (feature, threshold, compare);

    Return: an array of the prediction results of all the points in X
    '''
    if compare == '>':
        return np.sign(X[:, feature] - threshold)
    else:
        return np.sign(threshold - X[:, feature])

def bestDecisionStump(w, X, labels):
    '''
    Input: w: an array containing all the weights for training data
           X: an array of all training data, each row is one data point
           labels: an array containing the labels of training data
    Return: the best decision stump on the training data with the given weight
    '''
    N, d = X.shape
    minErr = N
    bestFeature = None
    bestThreshold = None
    bestOperator = None
    # brute-force search for best decision stump
    for feature in range(d):
        upperBound = np.max(X[:, feature])
        lowerBound = np.min(X[:, feature])
        threshold = lowerBound
        while threshold < upperBound:
            threshold += 0.05
            for compare in ['>', '<']:
                t = stumpClassificationResult(feature, threshold, compare, X)
                err = np.dot(w, np.not_equal(t, labels))
                if err < minErr:
                    minErr = err
                    bestFeature = feature
                    bestThreshold = threshold
                    bestOperator = compare
    return bestFeature, bestThreshold, bestOperator
# END OF HELPER FUNCTIONS


def adaBoost(X, labels, M):
    '''
    TO BE COMPLETED
    X: an array of all training data, each row is a data point
    labels: an array containing the labels of training results
    M: number of iterations

    return:
        functions: a list of decision stumps we get after iterations
        alpha: an array containing weights of classifiers in functions
    '''
    N = X.shape[0]
    w = 1.0/N * np.ones(N)
    alpha = np.zeros(M)
    functions = []
    for m in range(M):
        f = bestDecisionStump(w, X, labels)
        print('decision stump #%d: x_%d' % (m, f[0]),  f[2], f[1])
        functions.append(f)

        t = stumpClassificationResult(f[0], f[1], f[2], X)
        err = np.dot(w, np.not_equal(t, labels))
        alpha[m] = 0.5 * np.log((1-err)/(err))

        sum = 0
        for i in range(N):
            w[i] = w[i] * np.exp(-1 * labels[i] * alpha[m] * t[i])
            sum += w[i]
        w / sum
    return functions, alpha


def classify(functions, alpha, X):
    '''
    TO BE COMPLETED
    functions: a list of weak learners (decision stumps) we obtained in the
               training process
    alpha: an array containing the weights of weak classifiers in functions
    X: an array of input data, each row is a data point

    return:
        t: an array of {-1, 1} indicating classification results of X
    '''
    t_temp = []
    for m in range(len(functions)):
        t_temp.append(alpha[m] * stumpClassificationResult(functions[m][0], functions[m][1], functions[m][2], X))
    t_temp = np.transpose(t_temp)

    t = np.empty(shape=(t_temp.shape[0]))
    for i in range(t_temp.shape[0]):
        sum = 0
        for j in range(t_temp.shape[1]):
            sum += t_temp[i][j]
        t[i] = np.sign(sum)

    return t

def plot_graph():
    pos_examples_x1 = []
    pos_examples_x2 = []

    neg_examples_x1 = []
    neg_examples_x2 = []

    for i in range(X.shape[0]):
        if labels[i] == 1:
            pos_examples_x1.append(X[i][0])
            pos_examples_x2.append(X[i][1])
        else:
            neg_examples_x1.append(X[i][0])
            neg_examples_x2.append(X[i][1])

    plt.figure()
    plt.plot(pos_examples_x1, pos_examples_x2, 'xr')
    plt.plot(neg_examples_x1, neg_examples_x2, 'ob')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def main():
    # plot_graph()
    functions, alpha = adaBoost(X, labels, 2)
    print('alpha:', alpha)
    t = classify(functions, alpha, X)
    print(np.equal(t, labels))



if __name__ == '__main__':
    main()
