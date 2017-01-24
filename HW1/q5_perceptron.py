#!/usr/bin/env python

"""
EECS 445 - Introduction to Maching Learning
HW1 Q5 Perceptron Algorithm with Offset
"""

import numpy as np
import helper as h

def is_correct(xi, yi, theta, b):
    dot_prod = np.dot(xi, theta) + b
    if dot_prod == 0:
        return False
    return np.sign(dot_prod) == yi

def all_correct(X, y, theta, b):
    """
    Given (nxd) data matrix, (nx1) array y, (dx1) array theta and a scalar b,
    returns true if the classifier specified by theta and b
    correctly classified all examples
    """
    for n in range(X.shape[0]):
        if not is_correct(X[n], y[n], theta, b):
            return False
    return True


def perceptron(X, y):
    """
    Given (nxd) data matrix and(nx1) array y, implements the perceptron algorithm
    and return the classifier: theta, b and
    misclassfication array: alpha (an array where the i-th position
    has the number of times i-th point has been misclassified)
    """
    theta = np.zeros(X.shape[1], dtype=np.int)
    b = 0
    alpha = np.zeros(X.shape[0], dtype=np.int)
    while not all_correct(X, y, theta, b):
        for n in range(X.shape[0]):
            if not is_correct(X[n], y[n], theta, b):
                alpha[n]+=1
                theta += y[n]*X[n]
                b += y[n]

    return theta, b, alpha


def main(fname):
    X, y = h.load_data(fname)
    theta, b, alpha = perceptron(X, y)

    print("Done!")
    print("============== Classifier ==============")
    print("Theta: ", theta)
    print("b: ", b)

    print("\n============== Alpha ===================")
    print("i \t Number of Misclassifications")
    print("========================================")
    for i in range(len(alpha)):
        print("%d \t\t % d" % (i, alpha[i]))
    print("Total Number of Misclassifications: %d" % (np.sum(alpha)))

    return


if __name__ == '__main__':
    main("dataset/q5.csv")
