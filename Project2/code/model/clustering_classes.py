"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Clustering Classes - Skeleton
"""

import numpy as np
from scipy import stats


class Point(object):
    """
    Represents a data point
    """

    def __init__(self, name, label, original_attrs):
        """
        Initialize name, label and attributes
        """
        self.name = name
        self.label = label
        self.attrs = original_attrs

    def dimensionality(self):
        """Returns dimension of the point"""
        return len(self.attrs)

    def get_attrs(self):
        """Returns attr"""
        return self.attrs

    def distance(self, other):
        """
        other: point, to which we are measuring distance to
        Return Euclidean distance of this point with other
        """
        # TODO
        return np.linalg.norm(self.attrs - other.attrs)

    def get_name(self):
        """Returns name"""
        return self.name

    def get_label(self):
        """Returns label"""
        return self.label


class Cluster(object):
    """
    A Cluster is defined as a set of elements
    """

    def __init__(self, points):
        """
        Elements of a cluster are saved in a list, self.points
        """
        self.points = points

    def get_points(self):
        """Returns points in the cluster as a list"""
        return self.points

    def get_purity(self):
        """Returns number of points in cluster and the number of points
            with the most common label"""
        labels = []
        for p in self.points:
            labels.append(p.get_label())

        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)

    def get_centroid(self):
        """Returns centroid of the cluster"""
        # TODO
        centroid = np.zeros(shape=(self.points[0].dimensionality(),))
        for point in self.points:
            centroid = np.add(centroid, point.get_attrs())
        return Point('centroid', -1, np.divide(centroid, len(self.points)))

    def remove_point(self, point):
        """Remove given point from cluster"""
        self.points.remove(point)

    def equivalent(self, other):
        """
        other: Cluster, what we are comparing this Cluster to
        Returns true if both Clusters are equivalent, or false otherwise
        """
        if len(self.get_points()) != len(other.get_points()):
            return False
        matched = []
        for p1 in self.get_points():
            for point2 in other.get_points():
                if p1.distance(point2) == 0 and point2 not in matched:
                    matched.append(point2)
        if len(matched) == len(self.get_points()):
            return True
        else:
            return False


class ClusterSet(object):
    """
    A ClusterSet is defined as a list of clusters
    """

    def __init__(self):
        """
        Initialize an empty set, without any clusters
        """
        self.members = []

    def add(self, c):
        """
        c: Cluster
        Appends a cluster c to the end of the cluster list
        only if it doesn't already exist in the ClusterSet.
        If it is already in self.members, raise a ValueError
        """
        if c in self.members:
            raise ValueError
        self.members.append(c)

    def get_clusters(self):
        """Returns clusters in the ClusterSet"""
        return self.members[:]

    def get_centroids(self):
        """Returns centroids of each cluster in the ClusterSet as a list"""
        # TODO
        centroids = []
        for cluster in self.members:
            centroids.append(cluster.get_centroid())
        return centroids

    def get_score(self):
        """
            Returns accuracy of the clusering given by the clusters
            in the_cluster_set object
        """
        total_correct = 0
        total = 0
        for c in self.members:
            n, n_correct = c.get_purity()
            total = total + n
            total_correct = total_correct + n_correct

        return total_correct / float(total)

    def equivalent(self, other):
        """
        other: another ClusterSet object

        Returns true if both ClusterSets are equivalent, or false otherwise
        """
        if len(self.get_clusters()) != len(other.get_clusters()):
            return False

        matched = []
        for c1 in self.get_clusters():
            for c2 in other.get_clusters():
                if c1.equivalent(c2) and c2 not in matched:
                    matched.append(c2)
        if len(matched) == len(self.get_clusters()):
            return True
        else:
            return False

    def num_clusters(self):
        """Returns number of clusters in the ClusterSet"""
        return len(self.members)
