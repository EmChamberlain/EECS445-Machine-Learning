"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Homework 3
Script for running soft clustering using GMM on the MNIST dataset
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import string as s
from gmm import gmm
import matplotlib.pyplot as plt
from sklearn import datasets

"""Retrieve dataset from sklearn library"""
print("Loading data.....")
iris = datasets.load_iris()
trainX = iris.data
print("Shape of the input data: %d by %d" % (trainX.shape[0], trainX.shape[1]))

"""
GMM call with different values of number of clusters
	- num_K is an array containing the tested cluster sizes
	- cluster_proportions maps each cluster size to a size by 1 vector containing the mixture proportions
	- means is a dictionary mapping the cluster size to matrix of means
	- z_K maps each cluster size into a num_points by k matrix of pointwise cluster membership probabilities
	- sigma2 maps each cluster size to the corresponding sigma^2 value learnt
	- BIC_K contains the best BIC values for each of the cluster sizes
"""
print("We'll try different number of clusters with GMM and have multiple runs for each to identify the 'best' results")
num_K = [2, 3, 4, 5, 6, 7, 8]  # List of cluster sizes
means = {}  # Dictionary mapping cluster size to corresponding matrix of means
cluster_proportions = {}  # Dictionary mapping cluster size to corresponding mixture proportions vector
z_K = {}
sigma2 = {}  # Dictionary mapping cluster size to the learnt variance value
BIC_K = np.zeros([7, 1])
for idx in range(0, 7):
    # Running multiple runs (>= 10 times) for each cluster size
    k = num_K[idx]
    print("%d clusters...", k)
    for i in range(10):
        mu, pk, zk, si2, BIC = gmm(trainX, k)
        if BIC_K[idx] == 0 or BIC_K[idx] > BIC:
            means[k] = mu
            cluster_proportions[k] = pk
            z_K[k] = zk
            sigma2[k] = si2
            BIC_K[idx] = BIC

# Parameters learnt, now plot the BIC values and compute the means for the optimum cluster
print("GMM parameters learnt for each cluster size")
print("Plotting BIC vs. cluster sizes...")
min_bic_k_val = 0
min_bic = BIC_K[0]
for idx in range(1,7):
    if BIC_K[idx] < min_bic:
        minbic = BIC_K[idx]
        min_bic_k_val = idx
min_k_val = num_K[min_bic_k_val]
print("Optimal number of clusters: " + str(min_k_val))
print("Means: " + str(means[min_k_val]))
plt.plot(num_K, BIC_K, 'bo', label='BIC vs num_K')
plt.xlabel('num_K')
plt.ylabel('BIC')
plt.legend(loc='upper right')
plt.show()
