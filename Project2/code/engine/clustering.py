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

import random


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
    return random.sample(points, k)


def k_means_pp_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO
    starting_point = random.choice(points)
    centroids = [starting_point]
    for i in range(k-1):
        probabilities = np.ndarray(shape=(len(points), ))
        sum = 0
        for point_ind in range(len(points)):
            if points[point_ind] in centroids:
                probabilities[point_ind] = 0
            else:
                dist = np.power(points[point_ind].distance(centroids[-1]),2)
                probabilities[point_ind] = dist
                sum += dist
        probabilities = np.divide(probabilities, sum)
        next_point = np.random.choice(points, p=probabilities)
        centroids.append(next_point)
    return centroids


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
    init_list = None
    if init == 'random':
        init_list = random_init(points, k)
    elif init == 'cheat':
        init_list = k_means_pp_init(points, k)
    else:
        print('unknown init method: ' + str(init))
        exit(-1)

    init_assign = []
    for point in points:
        min_ind = 0
        min_dist = point.distance(init_list[0])
        for i in range(1, len(init_list)):
            if point.distance(init_list[i]) < min_dist:
                min_ind = i
                min_dist = point.distance(init_list[i])
        init_assign.append(min_ind)

    cluster_set = ccs.ClusterSet()
    for i in range(len(init_list)):
        cluster_set.add(ccs.Cluster([]))
    for i in range(len(points)):
        cluster_set.get_clusters()[init_assign[i]].get_points().append(points[i])

    centroids = cluster_set.get_centroids()
    while True:

        assign = []
        for  point in points:
            min_ind = 0
            min_dist = point.distance(centroids[0])
            for i in range(1, len(centroids)):
                if point.distance(centroids[i]) < min_dist:
                    min_ind = i
                    min_dist = point.distance(centroids[i])
            assign.append(min_ind)

        new_set = ccs.ClusterSet()
        for i in range(len(centroids)):
            new_set.add(ccs.Cluster([]))
        for i in range(len(points)):
            new_set.get_clusters()[assign[i]].get_points().append(points[i])

        if new_set.equivalent(cluster_set):
            break
        cluster_set = new_set


    return cluster_set




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

    plt.figure()
    plt.plot(k_vals, k_means_Scores, '--or', label='KMeans')
    plt.plot(k_vals, kpp_Scores, '--ob', label='KMeans++')
    plt.xlabel('Number of clusters, k', fontsize=16)
    plt.ylabel('Purity', fontsize=16)
    plt.legend(loc='upper left')
    plt.title('Part 2(c)')
    plt.show()

def main():
    X, y = clustering_utils.get_data()
    points = build_face_image_points(X, y)

    num_iterations = 10
    k_vals = list(range(1, 11))
    k_means_Scores = np.zeros(shape=(len(k_vals),num_iterations))
    kpp_Scores = np.zeros(shape=(len(k_vals),num_iterations))


    for i in range(num_iterations):
        print('Iteration: ' + str(i))
        for k_ind in range(len(k_vals)):
            k_means_Scores[k_ind][i] = k_means(points, k_vals[k_ind], init='random').get_score()
            kpp_Scores[k_ind][i] = k_means(points, k_vals[k_ind], init='cheat').get_score()

    plot_performance(np.mean(k_means_Scores, axis=1), np.mean(kpp_Scores, axis=1), k_vals)


if __name__ == '__main__':
    main()
