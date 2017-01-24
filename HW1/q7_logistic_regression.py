#!/usr/bin/env python

"""
EECS 445 - Introduction to Maching Learning
HW1 Q7 Logistic Regression
"""

import numpy as np
import helper as h


def newtons_method(X, y):
    """
    Given an nxd array X and an nx1 array y, finds the coefficients
    that fit the data using Newton's method using the
    stopping criterion: number of iteration == 100
    Returns theta, a (d+1)x1 array of the coefficients, including
    the offset/intercept term
    """
    theta = 0
    return theta


def main(fname):
    X, y = h.load_data(fname)
    theta = newtons_method(X, y)
    print("Theta: ", theta)
    print("Done!")


if __name__ == '__main__':
    main('dataset/q7.csv')
