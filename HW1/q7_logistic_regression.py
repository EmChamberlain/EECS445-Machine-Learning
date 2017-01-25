#!/usr/bin/env python

"""
EECS 445 - Introduction to Maching Learning
HW1 Q7 Logistic Regression
"""

import numpy as np
from numpy.linalg import pinv
import helper as h

def H(x, theta):
    exp = None
    try:
        exp = np.exp(np.dot(theta, x))
    except:
        return 1
    return 1 / (1 + (1/exp))

def generate_D(X, theta):
    d = X.shape[0]
    arr_D = np.ndarray(d)
    for i in range(d):
        arr_D[i] = -1*H(X[i], theta) * (1 - H(X[i], theta))
    return np.diag(arr_D)


def newtons_method(X, y):
    """
    Given an nxd array X and an nx1 array y, finds the coefficients
    that fit the data using Newton's method using the
    stopping criterion: number of iteration == 100
    Returns theta, a (d+1)x1 array of the coefficients, including
    the offset/intercept term
    """
    n = X.shape[0]
    d = X.shape[1]
    theta = np.zeros(d + 1)
    temp = np.ones((n, 1))
    X = np.append(X, temp, axis=1)
    #print(X)

    count = 0
    while count <= 100:
        for i in range(n):
            D = generate_D(X, theta)
            XT = np.transpose(np.matrix(X))
            #print(XT)
            XT_dot_D = np.dot(XT, D)
            #print(XT_dot_D)
            Hessian = np.dot(XT_dot_D, X)
            #print("Hessian: " + str(Hessian))
            H_inv = pinv(Hessian)
            #print(H_inv)
            H_Xi = H(X[i], theta)
            rhs = np.dot(X[i], (y[i] - H_Xi))
            #print("rhs: " + str(rhs))
            H_inv_dot_rhs = np.dot(H_inv, rhs).A1
            #print(H_inv_dot_rhs)
            theta -= H_inv_dot_rhs
        count += 1
    return theta


def main(fname):
    X, y = h.load_data(fname)
    theta = newtons_method(X, y)
    print("Theta: ", theta)
    print("Done!")


if __name__ == '__main__':
    main('dataset/q7.csv')
