"""
This module contains the NeuronNetwork class, which is to be used for quickly
generating and plotting visualizations of neuron networks.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import os
import sys
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
from matplotlib.pylab import plt

class NeuronNetwork(object):
    """
    This is a wrapper class for the NetworkX graph data structure.
    Use this class in order to conduct graph theory analysis on
    networks of neurons that were recorded using Calcium Imaging.
    """
    def __init__(self, dataframe):
        self.neurons = list(dataframe.columns)
        self.connections = get_corr_pairs(dataframe, p_val=0.01)
        self.network = self.create_graph(self.neurons, self.connections)
        self.mean_degree_centrality = self.compute_mean_degree_cent()
        self.connection_density = self.compute_connection_density()
        self.global_efficiency = nx.global_efficiency(self.network)
        self.clustering_coefficient = nx.average_clustering(self.network, weight="weight")
        self.max_clique_size = self.compute_max_clique_size()
        self.mean_clique_size = self.compute_mean_clique_size()
        self.mean_betw_centrality = self.compute_mean_betw_cent()
        self.avg_shortest_path_len = self.compute_avg_shortest_path_len()
        self.small_worldness = self.compute_small_worldness()

    def create_graph(self, nodes, edges):
        """Wrapper function for creating a NetworkX graph

        Each individual column of the provided DataFrame will be represented by
        a single node in the graph. Each pair of correlated nodes (neurons)
        will be connected by an edge, where the edge will receive a weight of
        the specific correlation coefficient of those two nodes.

        Args:
            dataframe: DataFrame

                a pandas DataFrame that contains the data to be represented
                with a NetworkX graph

            no_events_neurons: dictionary

        Returns:
            graph: NetworkX graph

                a graph of the neuronal network
        """
        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        for edge in edges:
            graph.add_edge(edge[0], edge[1], weight=round(edges[edge], 2))

        return graph

    def generate_random_graph(self):
        """Generates a random NetworkX graph based on the network of neurons.

        The specific amount of edges will be added between random pairs of nodes
        (neurons) in the graph.

        Returns:
            random_graph: NetworkX graph

                A NetworkX graph of the neuronal network with the same number of
                connections and neurons as the original network of neurons.
        """
        random_graph = nx.Graph()
        random_graph.add_nodes_from(self.neurons)

        weights = list()
        for edge in self.network.edges:
            weight = self.network.get_edge_data(edge[0], edge[1])["weight"]
            weights.append(weight)

        # Connect an E amount of random edges between all nodes in the random
        # graph.
        i = 0
        node_list = list(random_graph.nodes)
        while i < len(self.network.edges):
            end_node1 = node_list[np.random.randint(0, len(node_list))]
            end_node2 = node_list[np.random.randint(0, len(node_list))]

            # Make sure that the random edge generated does not already exist.
            if random_graph.get_edge_data(end_node1, end_node2) or random_graph.get_edge_data(end_node2, end_node1):
                continue

            random_index = np.random.randint(0, len(weights))
            random_graph.add_edge(end_node1, end_node2, weight=weights.pop(random_index))
            i += 1

        return random_graph

    def plot(self, **kwargs):
        """A wrapper function for plotting a NetworkX graph

        This function will draw a provided NetworkX graph using either the
        spring layout algorithm, or by the positions provided.

        Args:
            pos: dictionary, optional

                A dictionary of the network's neurons as keys and their (x ,y)
                coordinates as corresponding values.

            node_size: int, optional

                The size of the plotted neurons in the network.

            node_color: str, optional

                The color of the neurons to be plotted.

            fontsize: int, optional

                The size of the font in the plotted neurons.
        """

        # Get positions for all nodes
        pos = kwargs.get("pos", None)
        if pos is None:
            print("You did not provide a neuron position dictionary, "
                  + "so the Spring Layout algorithm will be used to "
                  + "plot the network", file=sys.stderr)
            pos = nx.spring_layout(self.network, weight="weight")

        # Size of the plot
        plt.figure(figsize=kwargs.get("figsize", (35, 35)))

        # nodes
        node_size = kwargs.get("node_size", 1000)
        color = kwargs.get("node_color", "pink")
        nx.draw_networkx_nodes(self.network, pos, node_size=node_size, node_color=color)

        edges, weights = zip(*nx.get_edge_attributes(self.network, "weight").items())

        # edges
        nx.draw_networkx_edges(self.network, pos, width=3.0, edge_color=weights, edge_cmap=plt.cm.YlGnBu)

        labels = nx.get_edge_attributes(self.network, "weight")
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=labels)

        # labels
        font_size = kwargs.get("font_size", 15)
        nx.draw_networkx_labels(self.network, pos, font_size=font_size, edge_labels=labels)

        plt.axis("off")

        save_to_file = kwargs.get("save", False)
        if save_to_file:
            title = kwargs.get("file_name", "Graph.png")
            file_format = kwargs.get("format", "PNG")
            plt.savefig(title, format=file_format)

        plt.show()

    def compute_connection_density(self):
        """Computes the connection density of a network of neurons.

        Connection density is the actual number of edges in the graph as a
        proportion of the total number of possible edges and is the simplest
        estimator of the physical cost — for example, the energy or other
        resource requirements — of a network. (Bullmore et al. 2009)

        total number of possible edges is: n(n-1) / 2,
        where n is the number of nodes (neurons) in the graph.
        """
        num_of_neurons = len(self.neurons)
        possible_num_of_edges = (num_of_neurons * (num_of_neurons-1)) / 2
        return len(list(self.network.edges())) / possible_num_of_edges

    def compute_mean_betw_cent(self):
        """Computes the mean betweeness centrality of a network of neurons.

           The centrality of a node measures how many of the shortest paths
           between all other nodes pairs in the network pass through it. A node
           with high centrality is thus crucial to efficient communication.
           (Bullmore et. al. 2009)

           https://en.wikipedia.org/wiki/Betweenness_centrality
        """
        graph_centrality = nx.betweenness_centrality(self.network, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_mean_degree_cent(self):
        """Computes the mean degree centrality of a network of neurons.

           The centrality of a node measures how many of the shortest paths
           between all other nodes pairs in the network pass through it. A node
           with high centrality is thus crucial to efficient communication.
           (Bullmore et. al. 2009)

           https://en.wikipedia.org/wiki/Centrality#Degree_centrality
        """
        graph_centrality = nx.degree_centrality(self.network)
        return np.mean(list(graph_centrality.values()))

    def compute_mean_eigen_cent(self):
        """Computes the mean Eigenvector centrality of a network of neurons.

            The centrality of a node measures how many of the shortest paths
            between all other nodes pairs in the network pass through it. A node
            with high centrality is thus crucial to efficient communication.
            (Bullmore et. al. 2009)

            https://en.wikipedia.org/wiki/Eigenvector_centrality
        """
        graph_centrality = nx.eigenvector_centrality(self.network, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_mean_katz_cent(self):
        """Computes the mean Katz centrality of a network of neurons.

           The centrality of a node measures how many of the shortest paths
           between all other nodes pairs in the network pass through it. A node
           with high centrality is thus crucial to efficient communication.
           (Bullmore et. al. 2009)

           https://en.wikipedia.org/wiki/Katz_centrality
        """
        graph_centrality = nx.katz_centrality(self.network, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_mean_load_cent(self):
        """Computes the mean load centrality of a network of neurons.

            The centrality of a node measures how many of the shortest paths
            between all other nodes pairs in the network pass through it. A node
            with high centrality is thus crucial to efficient communication.
            (Bullmore et. al. 2009)
        """
        graph_centrality = nx.load_centrality(self.network, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_max_clique_size(self):
        """Computes the size of the maxiumum clique in the network of neurons.

            A maximum clique of a graph, G, is a clique, such that there is no
            clique with more vertices.

            https://en.wikipedia.org/wiki/Clique_(graph_theory)#Definitions
        """
        return len(clique.max_clique(self.network))

    def compute_mean_clique_size(self):
        """Computes the mean clique size in the network of neurons.

        Finds all cliques in an undirected graph (network), and computes the
        mean size of all those cliques.

            Returns:
                mean: float

                    the mean clique size of the network of neurons.,
        """
        all_cliques = nx.enumerate_all_cliques(self.network)

        size = 0
        running_sum = 0
        for l in all_cliques:
            size += 1
            running_sum += len(l)

        mean = running_sum / size
        return mean

    def compute_avg_shortest_path_len(self):
        """Computes the average path shortest path length.

        This function compute the average path length L, as the average
        length of the shortest path connecting any paid of nodes in a
        network.

        Returns:
            avg_shortest_path_len: float

                The average shortest path length in the network of neurons.

        """
        graph = self.network
        shortest_path_lengths = list()
        node_list = list(graph.nodes)

        for i in range(0, len(node_list)):
            for j in range(i+1, len(node_list)):
                source = node_list[i]
                target = node_list[j]
                if not nx.has_path(graph, source, target):
                    continue

                shortest_path_lengths.append(nx.shortest_path_length(graph, source=source, target=target, weight="weight"))

        avg_shortest_path_len = np.mean(shortest_path_lengths)
        return avg_shortest_path_len

    def compute_small_worldness(self):
        small_worldness = self.clustering_coefficient/self.avg_shortest_path_len
        return small_worldness

def roll_worker(queue, dataframe, neuron_x, neuron_y, resamples):
    """Helper function for roll()

        Note: This function is only meant to be called by roll(), i.e.,
        this function should not used as a standalone function.

        Args:
            queue: Queue

                The blocking Queue to add the list of computed correlation
                coefficients that this function produces.

            dataframe: DataFrame

                A T x N matrix, where T := # of rows (observations) and
                N := # of neuron column vectors.

            neuron_x:

                The name of a neuron column vector in the provided dataframe.

            neuron_y:

                The name of a neuron column vector in the provided dataframe.

            resamples: int

                The amount of times to compute the test statistic, i.e.,
                correlation coefficient between the two provided neurons.

        Returns:
            corr_coefficients: list

                A list of all the computed correlation coefficients between
                the two provided neurons.
    """
    corr_coefficients = []
    high = len(dataframe.loc[:, neuron_x].index)

    time_series_x = dataframe.loc[:, neuron_x].values
    time_series_y = dataframe.loc[:, neuron_y].values
    for _ in range(resamples):
        time_series_x = np.roll(time_series_x, shift=np.random.randint(1, high))
        time_series_y = np.roll(time_series_y, shift=np.random.randint(1, high))
        corr_coefficients.append(np.corrcoef(time_series_x, time_series_y)[0][1])

    queue.put(corr_coefficients)

def roll(dataframe, neuron_x, neuron_y, resamples):
    """Create resampled distribution of correlation coefficients for two neurons

        Args:
            dataframe: DataFrame

                A T x N matrix, where T := # of rows (observations) and
                N := # of neuron column vectors.

            neuron_x:

                The name of a neuron column vector in the provided dataframe.

            neuron_y:

                The name of a neuron column vector in the provided dataframe.

            resamples: int

                The amount of times to compute the test statistic, i.e.,
                correlation coefficient between the two provided neurons.

        Returns:
            corr_coefficients: list

                A list of all the computed correlation coefficients between
                the two provided neurons.
    """
    resamples_per_worker = resamples // os.cpu_count()
    queue = Queue()
    processes = []
    rets = []

    for _ in range(os.cpu_count()):
        process = Process(target=roll_worker, args=(queue, dataframe, neuron_x, neuron_y, resamples_per_worker))
        processes.append(process)
        process.start()
    for process in processes:
        ret = queue.get()  # will block
        rets.append(ret)
    for process in processes:
        process.join()

    corr_coefficients = [item for sublist in rets for item in sublist]

    return corr_coefficients

def one_sided_p_val(actual_corrcoef, corr_coefficients):
    """Compute the one sided p-value of correlation coefficients.

        Args:
            actual_corrcoef: float

                The actual correlation coefficient of a pair of
                neurons.

            corr_coefficients: list

                The resampled distribution of all the possible correlation
                coefficients between two neurons.

        Returns:
            p_value: float

                The computed one sided p-value.
    """
    p = len(corr_coefficients)
    D = actual_corrcoef

    count = 0
    for D_i in corr_coefficients:
        if D_i >= D:
            count += 1

    p_value = (1 / p) * count

    return p_value

def get_corr_pairs(dataframe, **kwargs):
    """Find pairs of correlated neurons

        Goes through all possible pairs of neurons and
        saves the pairs that have statistically significant
        correlation coefficients.

        Args:
            dataframe: DataFrame

                A T x N matrix, where T := # of rows (observations) and
                N := # of neuron column vectors of all of the neurons.

            p_val: float, optional

                The p_val to use as the threshold for when to reject the null
                hypothesis, i.e., that the correlation coefficient of the pair
                of neurons are is statistically significant; default is 0.05

            resamples: int, optional

                The amount of times to roll the two neural time series, i.e.,
                the neural time series data from a given pair of neurons;
                default is 10000.

        Returns:

            corr_pairs: dictionary

                A dictionary that contains all of the statistically significant
                correlated pairs and their respective correlation coefficient,
                as such: <(neuron_x, neuron_y), corrcoef>
    """
    corr_pairs = {}
    p_val = kwargs.get("p_val", 0.05)
    resamples = kwargs.get("resamples", 10000)

    for i in range(1, len(dataframe.columns)+1):

        # Skip this iteration of i if the standard dev. is going to be 0
        if len(dataframe[i].unique()) == 1:
            continue

        for j in range(i+1, len(dataframe.columns)+1):

            # Skip this iteration if the standard dev. is going to be 0
            if len(dataframe[j].unique()) == 1:
                continue

            # Compute the real corrcoef between spiking data of two neurons
            actual_corrcoef = np.corrcoef(dataframe.loc[:, i], dataframe.loc[:, j])[0][1]

            # Skip this iteration if the real corrcoef is not positive
            if actual_corrcoef <= 0:
                continue

            # Carry out the rolling to build a resampled distribution
            resampled_vals = roll(dataframe, i, j, resamples)

            if one_sided_p_val(actual_corrcoef, resampled_vals) < p_val:
                corr_pairs[(i, j)] = actual_corrcoef

    return corr_pairs
