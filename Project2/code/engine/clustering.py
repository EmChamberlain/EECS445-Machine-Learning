"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Clustering - Skeleton
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import clustering_utils
import model.clustering_classes as ccs


def build_face_image_points(X, y):
    """
    Input:
        X : (n,d) feature matrix, in which each row represents an image
        y: (n,1) array, vector, containing labels corresponding to X
    Returns:
        List of Points
    """
    (n, d) = X.shape
    images = {}
    points = []
    for i in range(0, n):
        if y[i] not in images.keys():
            images[y[i]] = []
        images[y[i]].append(X[i, :])
    for face in images.keys():
        count = 0
        for im in images[face]:
            points.append(ccs.Point(str(face) + '_' + str(count), face, im))
            count = count + 1

    return points


def random_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO


def k_means_pp_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO


def k_means(points, k, init='random'):
    """
    Input:
        points: a list of Point objects
        k: the number of clusters we want to end up with
        init: The method of initialization, takes two valus 'cheat'
              and 'random'. If init='cheat', then use cheat_init to get
              initial clusters. If init='random', then use random_init
              to initialize clusters. Default value 'random'.

    Clusters points into k clusters using k_means clustering.

    Returns:
        Instance of ClusterSet corresponding to k clusters
    """
    # TODO


def plot_performance(k_means_Scores, kpp_Scores, k_vals):
    """
    Input:
        KMeans_Scores: A list of len(k_vals) average purity scores from running the
                       KMeans algorithm with Random Init
        KPP_Scores: A list of len(k_vals) average purity scores from running the
                    KMeans algorithm with KMeans++ Init
        K_Vals: A list of integer k values used to calculate the above scores

    Uses matplotlib to generate a graph of performance vs. k
    """
    # TODO


def main():
    X, y = clustering_utils.get_data()
    points = build_face_image_points(X, y)


if __name__ == '__main__':
    main()
