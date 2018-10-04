from sklearn import cluster

def affinity_propagation(similiarity_matrix):
    """Perform Affinity Propagation Clustering of data

    Note: This function is a wrapper for AffinityPropagation from scikit-learn
    Source: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html

    Args:
        similarity_matrix: DataFrame, shape (n_samples, n_samples)
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
        for neuron in list(similiarity_matrix.columns[labels==i]):
            clusters[neuron] = i

    return clusters

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
