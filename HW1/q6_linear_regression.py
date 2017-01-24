#!/usr/bin/env python

"""
EECS 445 - Introduction to Maching Learning
HW1 Q6 Linear Regression Optimization Methods)
Skeleton Code
Follow the instructions in the homework to complete the assignment.
"""
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import helper as h


def calculate_RMS_Error(X, y, theta, M):
    """
    Given nxd array X and nx1 array y, and (d x 1) theta specificying a
    (d-1)^th degree polynomial, calculates the root mean square error as defined
    in the assignment. Returns the error as a float.
    """
    n = X.shape[0]
    d = X.shape[1]
    phi = generate_polynomial_features(X, M)
    sum = 0
    for i in range(n):
        sum += np.power( (np.dot(theta, phi[i]) - y[i]) ,2)
    return np.sqrt(sum / n)

    return E_rms


def generate_polynomial_features(X, M):
    """
    Given an nx1 array X and an integer M, maps
    X to an M+1 dimensional feature vector e.g. [1, X, X^2, ...,X^M]
    Returns the mapped data as an nx(M+1) array.
    """
    Phi = np.ndarray(shape=(X.shape[0], M+1))
    for n in range(X.shape[0]):
        Phi[n, 0] = 1
        for m in range(1, M + 1):
            Phi[n, m] = np.power(X[n], M)
    return Phi

def calculate_error(phi, y, theta):
    sum = 0
    for i in range(phi.shape[0]):
        sum+= np.power(y[i] - np.dot(theta, phi[i]), 2)/2;
    return sum / phi.shape[0]

def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Given an nxd array X and an nx1 array y, finds the coefficients of a
    {d-1}^th degree polynomial that fit the data using least squares stochastic
    graident descent. Please do not shuffle your data points.
    Please use the stopping criteria: number of iteration < 1e5 or |new_error - prev_cost| < 1e-10
    Returns a dx1 array of the coefficients
    """

    count = 0
    new_error = 0
    prev_cost = 1
    theta = np.zeros(X.shape[1] + 1, dtype=np.float)
    phi = generate_polynomial_features(X, X.shape[1])
    while count < 1e5 and np.abs(new_error - prev_cost) >= 1e-10:
        # if count % 10000 == 0:
        #     print("Iteration: " + str(count))
        #     print("prev_cost: " + str(prev_cost))
        #     print("new_error: " + str(new_error))
        #     print("Difference: " + str(np.abs(new_error - prev_cost)))
        #     print("Theta: " + str(theta))
        prev_cost = calculate_error(phi, y, theta)
        for t in range(X.shape[0]):
            theta += (1/(count + 1))*(y[t]-np.dot(theta, phi[t]))*phi[t]
        new_error = calculate_error(phi, y, theta)
        count += 1
    print("Number of iterations: " + str(count))
    print("Error: " + str(calculate_error(phi, y, theta)))
    return theta


def closed_form_optimization(X, y, M, reg_param=0):
    """
    Given an nxd array X and an nx1 array y, finds the coefficients of a
    {d-1}^th degree polynomial that fit the data using the closed form solution
    discussed in class. reg_param is an optional regularization parameter
    Returns a dx1 array of the coefficients
    """
    n = X.shape[0]
    d = X.shape[1] + 1
    phi = generate_polynomial_features(X, M)
    b = 0
    A = np.zeros((M + 1, M + 1))
    for i in range(n):
        phi_mat = np.matrix(phi[i])
        b -= y[i]*phi[i]
        A += np.transpose(phi_mat)*phi_mat

    b = np.divide(b, n)
    A = np.divide(A, n)
    Both_inv = pinv(reg_param * np.identity(M + 1) + A)
    return np.dot(Both_inv, b)


def part_a(fname_train):
    """
    This function should contain all the code you implement to complete (a)
    """
    print("========== Part A ==========")
    X, y = h.load_data(fname_train)
    learning_rate = 1e-4
    theta = ls_stochastic_gradient_descent(X, y, learning_rate)
    print("Descent: " + str(theta))

    theta = closed_form_optimization(X, y, X.shape[1])
    print("Closed form: " + str(theta))
    print("Done!")

    return


def part_b(fname_train, fname_test):
    """
    This function should contain all the code you implement to complete (b)
    """
    print("=========== Part B ==========")
    X_train, y_train = h.load_data(fname_train)
    X_test, y_test = h.load_data(fname_test)
    RMS_data_train = np.ndarray(shape=(11))
    RMS_data_test = np.ndarray(shape=(11))
    M_data = np.ndarray(shape=(11))
    for m in range(11):
        M_data[m] = m
        theta = closed_form_optimization(X_train, y_train, m)
        RMS_data_train[m] = calculate_RMS_Error(X_train, y_train, theta, m)
        RMS_data_test[m] = calculate_RMS_Error(X_test, y_test, theta, m)
    plt.plot(M_data, RMS_data_train, 'ro')
    plt.plot(M_data, RMS_data_test, 'go')
    plt.ylabel("RMS")
    plt.xlabel("M")
    plt.show()
    print("Done!")

    return


def part_c(fname_train, fname_test):
    """
    This function should contain all the code you implement to complete (c)
    """
    print("=========== Part C ==========")
    param_arr = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    M = 10
    X_train, y_train = h.load_data(fname_train)
    X_test, y_test = h.load_data(fname_test)
    RMS_data_train = np.ndarray(shape=(10))
    RMS_data_test = np.ndarray(shape=(10))
    L_data = np.ndarray(shape=(10))

    for l in range(len(param_arr)):
        L_data[l] = param_arr[l]
        theta = closed_form_optimization(X_train, y_train, M, param_arr[l])
        RMS_data_train[l] = calculate_RMS_Error(X_train, y_train, theta, M)
        RMS_data_test[l] = calculate_RMS_Error(X_test, y_test, theta, M)

    plt.plot(L_data, RMS_data_train, 'ro')
    plt.plot(L_data, RMS_data_test, 'go')
    plt.ylabel("RMS")
    plt.xlabel("lambda")
    plt.show()
    print("Done!")
    return


def main(fname_train, fname_test):
    part_a(fname_train)
    part_b(fname_train, fname_test)
    part_c(fname_train, fname_test)

    return


if __name__ == '__main__':
    main("dataset/q6_train.csv", "dataset/q6_test.csv")
