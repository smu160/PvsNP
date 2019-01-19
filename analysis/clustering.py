"""
This module contains all the functions necessary for clustering, computing
similarity measures, and functional connectivity (edge weights in a graph).

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import itertools

from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import numpy as np


def compute_connections(dataframe, similarity_measure=normalized_mutual_info_score):
    """Create a dict of node pairs and their corresponding "edge weights"

    Compute a measure of similarity between `n choose 2` variables

    Args:
        dataframe: pandas DataFrame
            An m x n DataFrame, where n is the number of columns (variables),
            and m is the number of rows (observations).

        similarity_measure: function, optional, default: sklearn.metrics.normalized_mutual_info_score
            A function that quantifies the similarity between data points.

    Returns:
        connections: dict
            A dictionary of <(v_i, v_j): corrcoef> key-value pairs.
    """
    matrix = similarity_matrix(dataframe, similarity_measure=similarity_measure)
    connections = {(i, j): matrix[i][j] for i, j in itertools.combinations(matrix.columns, 2)}

    return connections


def similarity_matrix(dataframe, similarity_measure=normalized_mutual_info_score):
    """Create a similarity matrix between data points.

    Args:
        dataframe: pandas DataFrame
            An m x n DataFrame, where n is the number of columns (variables),
            and m is the number of rows (observations).

        similarity_measure: function, optional, default: sklearn.metrics.normalized_mutual_info_score
            A function that quantifies the similarity between data points.

    Returns:
        matrix: pandas DataFrame
            An n x n DataFrame, where each component, a_{ij}, is the
            quantified similarity between variable v_{i} and v_{j}.
    """
    rows = [dataframe.apply(similarity_measure, args=(dataframe[col],)) for col in dataframe]
    matrix = pd.DataFrame(rows, index=dataframe.columns)

    return matrix


def compute_corrcoef(dataframe, threshold=0.0):
    """Create dict of node pairs & their corresponding correlation coefficients

    Args:
        dataframe: pandas DataFrame
            An m x n DataFrame, where n is the number of columns (variables),
            and m is the number of rows (observations).

        threshold: float, optional, default: 0.0
            The cutoff value for any correlation coefficient, r, such that two
            variables, (v_i, v_j) will only be appended to the dictionary of
            connections if r >= threshold.

    Returns:
        connections: dict
            A dictionary of <(v_i, v_j): corrcoef> key-value pairs.
    """
    connections = {}

    for v_i, v_j in itertools.combinations(dataframe.columns, 2):
        if dataframe[v_i].std() == 0 or dataframe[v_j].std() == 0:
            continue

        corrcoef = np.corrcoef(dataframe[v_i].values, dataframe[v_j].values)[0][1]
        if corrcoef >= threshold:
            connections[(v_i, v_j)] = corrcoef

    return connections


def affinity_propagation(similiarity_matrix):
    """Perform Affinity Propagation Clustering of data

    Note: This function is a wrapper for AffinityPropagation from scikit-learn

    Source: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html

    Args:
        similarity_matrix: pandas DataFrame, shape (n_samples, n_samples)
            Matrix of similarities between points.

    Returns:
        clusters: dictionary
            A dictionary of <sample: cluster label> key-value pairs.
    """
    clusters = {}

    labels = cluster.AffinityPropagation().fit_predict(similiarity_matrix)
    n_labels = labels.max()

    clusters = {}
    for i in range(n_labels+1):
        for neuron in list(similiarity_matrix.columns[labels == i]):
            clusters[neuron] = i

    return clusters


# TODO: Use default dict
def extract_clusters(clusters):
    """Extract all the clusters into a dictionary of lists.

    This function takes in a dictionary of <neuron: cluster label>
    key-value pairs and produces a dictionary of <cluster label: neurons>

    Args:
        clusters: dictionary
            A dictionary of <sample: cluster label> key-value pairs.

    Returns:
        extracted_clusters: dictionary
            A dictionary of <cluster label: neurons list> key-value pairs.
    """
    extracted_clusters = {}

    for neuron, cluster_label in clusters.items():
        if extracted_clusters.get(cluster_label, None):
            extracted_clusters[cluster_label].append(neuron)
        else:
            extracted_clusters[cluster_label] = [neuron]

    return extracted_clusters
