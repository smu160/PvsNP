"""
This module contains the NeuronNetwork class, which is to be used for quickly
generating and plotting visualizations of neuron networks.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import sys
import itertools
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
from matplotlib.pylab import plt

class NeuronNetwork:
    """
    This is a wrapper class for the NetworkX graph data structure. Use this
    class in order to conduct graph theory analysis on networks of neurons
    that were observed during experiments/trials.
    """

    def __init__(self, neurons, connections):
        self.neurons = list(neurons)
        self.connections = connections
        self.network = self.create_graph(self.neurons, self.connections)

    def create_graph(self, nodes, edges):
        """Wrapper function for creating a NetworkX graph.

        Each individual column of the provided DataFrame will be represented by
        a single node in the graph. Each pair of correlated nodes (neurons)
        will be connected by an edge, where the edge will receive a weight of
        the specific correlation coefficient of those two nodes.

        Args:
            nodes: list
                A list of all the neurons/nodes in the network/graph.

            edges: dict {(int, int): scalar, ..., (int, int): scalar}
                A dictionary of key-value pairs, where each key is a tuple
                representing an edge between two neurons/nodes, and each value
                is a scalar value representing the weight of that edge.

        Returns:
            graph: NetworkX graph
                A graph of the neuronal network.
        """
        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        for edge in edges:
            graph.add_edge(edge[0], edge[1], weight=edges.get(edge, None))

        return graph

    def plot(self, **kwargs):
        """A wrapper function for plotting a NetworkX graph

        This function will draw a provided NetworkX graph using either the
        spring layout algorithm, or by the positions provided.

        Args:
            pos: dict, optional, default: networkx.drawing.layout.spring_layout
                A dictionary of the network's neurons as keys and their (x, y)
                coordinates as corresponding values.

            node_size: int, optional
                The size of the plotted neurons in the network.

            node_colors: list, optional
                The colors of the neurons to be plotted.

            fontsize: int, optional, default: 10
                The size of the font labelling the plotted neurons.

            draw_edges: bool, optional, default: True
                When True, the edges between all nodes will be drawn. When
                False, the edges between all nodes will be omitted from the
                figure.

            save: bool, optional, default: False
                When True, the plotted figure will be saved to the current
                working directory with its title as the file name, in PDF
                format.

            title: str, optional, default: None
                The title of the plotted graph/network.

            a: float, optional
                The transparency of the nodes
                
            label_nodes: bool, optional
                Whether to display node labels. Default is True
                
            edge_a: float, optional
                The transparency of the edges. Default is 1
            
            edge_weight: float, optional
                The width of the edge. Default is 2
                
            label_edges: bool, optional
                Whether to display edge labels. Default is False.

        Returns:
            pos: dict
                A dictionary of the network's neurons as keys and their (x, y)
                coordinates as corresponding values.
        """

        # Get positions for all nodes
        pos = kwargs.get("pos", None)
        if pos is None:
            print("You did not provide a neuron position dictionary. The spring layout function will be used to plot the network", file=sys.stderr)
            pos = nx.spring_layout(self.network, weight="weight")

        # Size of the plot
        figsize = kwargs.get("figsize", (30, 30))
        plt.figure(figsize=figsize)

        # Nodes
        node_size = kwargs.get("node_size", 600)
        node_colors = kwargs.get("node_colors", self.neurons)
        a = kwargs.get('a',.5)
        disp_node_labels=kwargs.get('label_nodes',True)
        nx.draw_networkx_nodes(self.network, pos, alpha=a, node_size=node_size, cmap=plt.cm.Dark2, node_color=node_colors)

        _, weights = zip(*nx.get_edge_attributes(self.network, "weight").items())

        # Draw edges
        edgeweight=kwargs.get('edgeweight',2)
        edge_a = kwargs.get('edge_a',1)
        edge_color = kwargs.get('edgecolor',weights)
        if kwargs.get("draw_edges", True):
            nx.draw_networkx_edges(self.network, pos, alpha=edge_a, width=edgeweight,edge_color=edge_color)
            
        # Labels
        
        label_edges=kwargs.get("label_edges",False)
        if label_edges:
            font_size = kwargs.get("font_size", 10)
            nx.draw_networkx_labels(self.network, pos, font_size=font_size)

        title = kwargs.get("title", None)
        plt.title(title)
        plt.axis("off")

        save_to_file = kwargs.get("save", False)
        if save_to_file:
            plt.savefig(title, format="pdf", dpi=600)

        plt.show()
        return pos

    def compute_connection_density(self):
        """Computes the connection density of a network of neurons.

            Connection density is the actual number of edges in the graph as a
            proportion of the total number of possible edges and is the simplest
            estimator of the physical cost — for example, the energy or other
            resource requirements — of a network. (Bullmore et al. 2009)

            Total number of possible edges is: n(n-1)/2,
            where n is the number of nodes (neurons) in the graph.
        """
        num_of_neurons = len(self.neurons)
        possible_num_of_edges = (num_of_neurons * (num_of_neurons-1)) / 2
        return len(list(self.network.edges())) / possible_num_of_edges

    def mean_betw_cent(self, weight="weight"):
        """Computes the mean betweeness centrality of a network of neurons.

            The centrality of a node measures how many of the shortest paths
            between all other nodes pairs in the network pass through it. A node
            with high centrality is thus crucial to efficient communication.
            (Bullmore et. al. 2009)

            https://en.wikipedia.org/wiki/Betweenness_centrality
        """
        betw_centrality = nx.betweenness_centrality(self.network, weight=weight)
        return np.mean(list(betw_centrality.values()))

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
                The mean clique size of the network of neurons.
        """
        all_cliques = nx.enumerate_all_cliques(self.network)

        size = 0
        running_sum = 0
        for cliq in all_cliques:
            size += 1
            running_sum += len(cliq)

        mean = running_sum / size
        return mean

    def avg_shortest_path_len(self, weight="weight"):
        """Computes the average path shortest path length.

        This function computes the average path length L, as the average
        length of the shortest path connecting any paid of nodes in a
        network.

        Args:
            weight: str or None, optional, default: None
                If None, every edge has weight/distance/cost 1. If a string, use
                this edge attribute as the edge weight. Any edge attribute not
                present defaults to 1.

        Returns:
            avg_shortest_path_len: float
                The average shortest path length in the network of neurons.

        """
        graph = self.network
        node_list = self.neurons
        shortest_path_lengths = []

        for node_pair in itertools.combinations(node_list, 2):
            source = node_pair[0]
            target = node_pair[1]
            if not nx.has_path(graph, source, target):
                continue

            shortest_path_lengths.append(nx.shortest_path_length(graph, source=source, target=target, weight=weight))

        avg_shortest_path_len = np.mean(shortest_path_lengths)
        return avg_shortest_path_len

    def small_worldness(self, weight="weight"):
        """Computes the small worldness of the neuron network.

        Returns:
            small_worldness: float
                The clustering coefficient divided by the average shortest path
                length of the neuron network.
        """
        cluster_coeff = nx.average_clustering(self.network, weight=weight)
        avg_shortest_path_len = self.avg_shortest_path_len(weight=weight)
        small_worldness = cluster_coeff / avg_shortest_path_len
        return small_worldness
