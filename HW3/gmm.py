"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Homework 3
The gmm function takes in as input a data matrix X and number of gaussians in the mixture model
The implementation assumes that the covariance matrix is shared and is a spherical diagonal covariance matrix
You have to fill in the pieces whereever you see ????
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.misc import logsumexp


def gmm(trainX, num_K, num_iter=20):
    """
        input trainX is a N by D matrix containing N datapoints, num_K is the number of clusters or mixture components desired.
        num_iter is the maximum number of EM iterations run over the dataset
        For the output:
            - mu which is K by D, the coordinates of the means
            - pk, which is K by 1 and represents the cluster proportions
            - zk, which is N by K, has at each z(n,k) the probability that the nth data point belongs to cluster k, specifying the cluster associated with each data point
            - si2 is the estimated (shared) variance of the data
            - BIC is the Bayesian Information Criterion (smaller BIC is better)
    """
    N = trainX.shape[0]
    D = trainX.shape[1]

    try:
        if num_K >= N:
            raise AssertionError
    except AssertionError:
        print("You are trying too many clusters")
        raise

    si2 = 1  # Initialization of variance
    pk = np.ones([num_K, 1])/num_K  # Uniformly initialize cluster proportions
    mu = np.random.randn(num_K, D)  # Random initialization of clusters

    zk = np.zeros([N, num_K])  # Matrix containing cluster membership probability for each point

    for iter in range(0, num_iter):
        """
            E-Step
            In the first step, we find the expected log-likelihood of the data which is equivalent to:
            finding cluster assignments for each point probabilistically
            In this section, you will calculate the values of zk(n,k) for all n and k according to current values of si2, pk and mu
        """


        for i in range(N):
            for j in range(num_K):
                exp_arr = np.zeros([num_K, 1])
                for k in range(num_K):
                    exp_arr[k] += np.log(pk[k])
                    # np.exp(-(np.power(trainX[i] - mu[k], 2)) / (2*si2)) / (np.sqrt(2 * np.pi * si2))
                    # print('***********')
                    # print(trainX[i])
                    # print(mu[k])
                    # print(si2)
                    # print('***********')
                    exp_arr[k] += mvn.logpdf(trainX[i], mu[k], si2*np.eye(D))
                b = np.exp(logsumexp(exp_arr))
                a = pk[j]*mvn.pdf(trainX[i], mu[j], si2*np.eye(D))
                zk[i][j] = a/b
        """
            M-step
            Compute the GMM parameters from the expressions which you have in your writeup
        """

        # Estimate new value of pk
        for k in range(num_K):
            sum = 0
            for i in range(N):
                sum += zk[i][k]
            pk[k] = sum / N

        # Estimate new value for mu
        for k in range(num_K):
            n_hat = 0
            sum_zik_x = 0
            for i in range(N):
                n_hat += zk[i][k]
                sum_zik_x += zk[i][k]*trainX[i]
            mu[k] = sum_zik_x / n_hat


        # Estimate new value for sigma^2
        sum = 0
        for i in range(N):
            for k in range(num_K):
                sum += zk[i][k]*np.dot( np.transpose(trainX[i] - mu[k]) , trainX[i] - mu[k])
        si2 = sum / (N*D)

    # Computing the expected likelihood of data for the optimal parameters computed
    maximum_log_likelihood = 0
    for i in range(N):
        max_log_exp_arr = np.zeros([num_K,1])
        for k in range(num_K):
            max_log_exp_arr[k] += np.log(pk[k])
            max_log_exp_arr[k] += mvn.logpdf(trainX[i], mu[k], si2*np.eye(D))
        maximum_log_likelihood += logsumexp(max_log_exp_arr)
    print(maximum_log_likelihood)
    # Compute the BIC for the current cluster using the expected log-likelihood
    BIC = (num_K*(D + 1))*np.log(N) - 2*maximum_log_likelihood
    # BIC = maximum_log_likelihood - (num_K*(D + 1))*(1/2)*np.log(N)

    return mu, pk, zk, si2, BIC
