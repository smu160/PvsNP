"""
This module contains the NeuronNetwork class, which is to be used for quickly
generating and plotting visualizations of neuron networks.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
from analysis.analysis_utils import FeatureExtractor

class NeuronNetwork(object):
    """
    This is a wrapper class for the NetworkX graph data structure.
    Use this class in order to conduct graph theory analysis on
    networks of neurons that were recorded using Calcium Imaging.
    """
    def __init__(self, dataframe):
        self.network = self.create_graph(dataframe)
        self.neurons = list(self.network.nodes())
        self.mean_degree_centrality = self.compute_mean_degree_cent()
        self.connection_density = self.compute_connection_density()
        self.global_efficiency = nx.global_efficiency(self.network)
        self.local_efficiency = nx.local_efficiency(self.network)
        self.clustering_coefficient = nx.average_clustering(self.network, weight="weight")
        self.max_clique_size = self.compute_max_clique_size()
        self.mean_clique_size = self.compute_mean_clique_size()
        # self.mean_betw_centrality = self.compute_mean_betw_cent()
        # self.mean_katz_centrality = self.compute_mean_katz_cent()
        # self.mean_load_centrality = self.compute_mean_load_cent()

    def create_graph(self, dataframe):
        """Wrapper function for creating a NetworkX graph

        Each individual column of the provided DataFrame will be represented by
        a single node in the graph. Each pair of correlated nodes (neurons)
        will be connected by an edge, where the edge will receive a weight of
        the specific correlation coefficient of those two nodes.

        Args:
            dataframe: DataFrame

                a pandas DataFrame that contains the data to be represented
                with a NetworkX graph

        Returns:
            graph: NetworkX graph

                a graph of the neuronal network
        """
        graph = nx.Graph()
        graph.add_nodes_from(dataframe.columns)
        corr_pairs = FeatureExtractor.find_correlated_pairs(dataframe, correlation_coeff=0.3)

        for key in corr_pairs:
            graph.add_edge(key[0], key[1], weight=round(corr_pairs[key], 3))

        return graph

    def gen_rand_graph(self, dataframe):
        """Generates a random NetworkX graph

        Each individual column of the provided DataFrame will be represented by
        a single node in the graph. The amount of correlated nodes (neurons) in
        the provided DataFrame will be computed, and that specific amount of
        edges will be added between random pairs of nodes (neurons) in the graph.

        Args:
            dataframe: DataFrame

                the pandas DataFrame to use as a basis for the random graph

        Returns:
            graph: NetworkX graph
                a NetworkX graph of the neuronal network
        """
        graph = nx.Graph()
        graph.add_nodes_from(dataframe.columns)
        corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

        # Connect a len(correlated_pairs_dict) amount of random edges between
        # all nodes in the random graph
        end_node1 = np.random.randint(1, len(dataframe.columns)+1)
        end_node2 = np.random.randint(1, len(dataframe.columns)+1)
        for _ in range(len(corr_pairs)):
            graph.add_edge(end_node1, end_node2)

        return graph

    def plot_with_spring_layout(self, **kwargs):
        """A wrapper function for plotting a NetworkX graph

        This function will draw a provided NetworkX graph using the spring
        layout algorithm.

        Args:
            node_size: int, optional

                the size of the plotted neurons in the network.

            node_color: str, optional

                the color of the neurons to be plotted.

            fontsize: int, optional

                the size of the font in the plotted neurons.
        """

        # positions for all nodes
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

    def plot_by_true_pos(self):
        """Plots the network of neurons as they look in reality.
        """
        pass

    # TODO: Find out if this is still needed and fix this function
    def generate_random_graph(self):
        """Generates a random graph based on the graph
        """
        # positions for all nodes
        pos = nx.spring_layout(self.network, weight='weight')

        plt.figure(figsize=(15, 15))

        # nodes
        nx.draw_networkx_nodes(self.network, pos, node_size=700, node_color='lightblue')

        # edges
        nx.draw_networkx_edges(self.network, pos, width=1.0)

        labels = nx.get_edge_attributes(self.network, 'weight')
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=labels)

        # labels
        nx.draw_networkx_labels(self.network, pos, font_size=15, edge_labels=labels)

        plt.axis('off')
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
