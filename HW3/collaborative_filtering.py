# -*- coding: utf-8 -*-
"""
EECS 445 
Homework 3
Collaborative Filtering
"""
import numpy as np
import scipy.io as sio
import cvxopt


def gram_matrix(X):
    """Computes the kernel matrix for SVM, based on a linear kernel
    Input: X an nxd array
    Output: K an nxn array
    """
    return np.dot(X, X.T)


def svm_linear(X, y, C):
    """Finds a soft-margin SVM *without* offset
    Input: X an nxd array of examples/features
           y an nx1 array of labels
           C a scalar cost parameter
    Output: theta a 1xd array, representing the learned coefficients
    """
    n, n_features = X.shape
    K = gram_matrix(X)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n))
    lb = cvxopt.matrix(np.zeros([n, 1]))
    ub = cvxopt.matrix(np.ones(n) * C)
    G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = cvxopt.matrix(np.vstack([-lb, ub]))
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, None, None)
    lagrange_multipliers = np.ravel(solution['x'])

    sv_indices = lagrange_multipliers > max(lagrange_multipliers) * 1e-8
    alphas = lagrange_multipliers[sv_indices]
    sv_x = X[sv_indices]
    sv_labels = y[sv_indices]
    n, d = sv_x.shape
    theta = np.zeros([1, d])
    for dim in range(d):
        theta[:, dim] = np.dot(sv_labels * alphas, sv_x[:, dim])
    return theta


def svm_cf(Y, d, C):
    """Solves a matrix factorization problem Y approx U V' iteratively using 
    linear SVMs with slack (parameter C)
    Input: Y is nxm matrix
           d is an integer
           C is the cost parameter of the SVM
    Output:
        U is an nxd array
        V is an mxd array 
    """

    n, m = Y.shape
    # initialize guess using SVD
    [U, S, V] = np.linalg.svd(Y, full_matrices=0)
    V = np.transpose(V)
    I = range(0, d)
    D = np.diag(np.sqrt(S[I]))
    U = np.dot(U[:, I], D)
    V = np.dot(V[:, I], D)

    X = np.dot(U, np.transpose(V))  # resulting initial guess about matrix

    niter = 1
    while True:
        # optimize U
        for i in range(n):
            J = np.nonzero(Y[i, :])
            if len(J) > 0:
                U[i, :] = svm_linear(V[J[0], :], np.transpose(Y[i, J[0]]), C)
            else:
                U[i, :] = 0

        # optimize V
        for j in range(m):
            I = np.nonzero(Y[:, j])
            if len(I) > 0:
                V[j, :] = svm_linear(U[I[0], :], np.transpose(Y[I[0], j]), C)
            else:
                V[j, :] = 0

        Xnew = np.dot(U, np.transpose(V))

        # stop if sign predictions didn't change in one iteration
        if sum(sum(np.sign(X) != np.sign(Xnew))) == 0:
            break
        X = Xnew
        niter = niter + 1

    Yhat = np.sign(X)
    I = np.nonzero(Y)  # non-zero elements
    # print('Number of non-zero elements in Y:'+str(len(Yhat[I])))

    training_err = sum(Yhat[I] != Y[I])
    # print('Number of U-V iterations: '+str(niter))
    # print('Training error: '+str(training_err))

    return U, V


def load_data():
    """ Loads movie data from file
    Output: Y an nxm array of user/movie ratings (sparse matrix)
            T an nxm array of user/movie ratings (ground truth)
    """
    contents = sio.loadmat('movie.mat')
    Y = contents['Y']
    T = contents['T']
    return Y, T


# If you need to define any helper functions, please include them below
# this remark

if __name__ == "__main__":
    """ 
    Complete for HW3 part f
    """
    Y, T = load_data()

    r_vals = [20, 50, 100]
    d_vals = [1, 2, 5]
    C = 1
    for r in r_vals:
        for d in d_vals:
            U, V = svm_cf(Y, d, C)
            Y_hat = np.dot(U, np.transpose(V))
            for i in range(Y_hat.shape[0]):
                for j in range(Y_hat[i].shape[0]):
                    if Y_hat[i][j] > 0:
                        Y_hat[i][j] = 1
                    elif Y_hat[i][j] < 0:
                        Y_hat[i][j] = -1
                    else:
                        Y_hat[i][j] = 0
            training_correct = 0
            training_wrong = 0
            test_correct = 0
            test_wrong = 0
            for i in range(r):
                for j in range(T[i].shape[0]):
                    if Y[i][j] != 0:
                        if Y[i][j] == Y_hat[i][j]:
                            training_correct += 1
                        else:
                            training_wrong += 1
                    if Y[i][j] == 0 and T[i][j] != 0:
                        if T[i][j] == Y_hat[i][j]:
                            test_correct += 1
                        else:
                            test_wrong += 1

            training_error = training_wrong / (training_correct + training_wrong)
            test_error = test_wrong / (test_correct + test_wrong)
            print('r = ' + str(r) + ', d = ' + str(d) + ': training = ' + str(training_wrong) + '/' + str(
                training_correct + training_wrong) + '; test = ' + str(test_wrong) + '/' + str(
                test_correct + test_wrong))
