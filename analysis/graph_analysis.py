#
# PvsNP: toolbox for reproducible analysis & visualization of neurophysiological data.
# Copyright (C) 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
This module contains the NeuronNetwork class and its respective functions
for generating networks (the graph theory kind) of observed neurons, plotting
visualizations of the networks, as well as computing various network measures.
"""

__author__ = "Saveliy Yusufov"
__date__ = "1 March 2019"
__license__ = "GPL"
__maintainer__ = "Saveliy Yusufov"
__email__ = "sy2685@columbia.edu"

import sys
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
from matplotlib.pylab import plt


class NeuronNetwork:
    """
    This is a wrapper class for the NetworkX graph data structure for conducting
    graph theoretical analysis on networks of neurons that were observed during
    experiments/trials.
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

            nodelist: list, optional, default: self.neurons
                Draw only specified neurons (nodes).

            node_size: int or list, optional, default: 300
                The size of the plotted neurons in the network.

            node_color: (color string, or array of floats), optional, default: 'r'
                Can either be a single color format string (default=’r’), or a
                sequence of colors with the same length as nodelist. If numeric
                values are specified they will be mapped to colors using the
                cmap and vmin, vmax parameters. See matplotlib.scatter for more
                details.

            cmap: Matplotlib colormap, optional, default: None
                Colormap for mapping intensities of nodes.

            alpha: float, optional, default: 1.0
                The transparency of the nodes.

            node_borders: (None, scalar, or sequence), optional, default: 'black'
                The color(s) of the node borders.

            edgelist: (collection of edge tuples), optional, default: self.connections
                Draw only specified edges. By default, the edges between all
                nodes will be drawn. If `[]` (i.e. empty list), then the edges
                between all nodes will be omitted from the figure.

            edge_alpha: float, optional, default is 1.0
                The transparency of the edges.

            edge_color: (color string, or array of floats), default: 'r'
                Can either be a single color format string, or a sequence of
                colors with the same length as edgelist. If numeric values are
                specified they will be mapped to colors using the edge_cmap and
                edge_vmin, edge_vmax parameters.

            edge_cmap : Matplotlib colormap, optional, default: None
                Colormap for mapping intensities of edges.

            width: float, optional, default: 1.0
                The width of the edges.

            labels: bool, optional, default: True
                If False, then omit node labels and edge labels.

            font_size: int, optional, default: 10
                The size of the font for text labels.

            figsize: tuple, optional, default: (20, 20)
                The size of the network figure to be plotted.

            savefig: bool, optional, default: False
                When True, the plotted figure will be saved to the current
                working directory, in PDF format, at the default (or specified)
                DPI.

            dpi: int, optional, default: 600
                The amount of dots per inch to use when saving the figure. In
                accordance with Nature's guidelines, the default is 600.
                Source: https://www.nature.com/nature/for-authors/final-submission

            title: str, optional, default: None
                The title of the plotted graph/network.

        Returns:
            pos: dict
                A dictionary of the network's neurons as keys and their (x, y)
                coordinates as corresponding values.
        """

        # Get positions for all nodes
        pos = kwargs.get("pos", None)
        if pos is None:
            print("A neuron position dictionary was not provided! The spring_layout function will be used to plot the network.", file=sys.stderr)
            pos = nx.spring_layout(self.network, weight="weight")

        # Size of the plot
        plt.figure(figsize=kwargs.get("figsize", (20, 20)))

        # Nodes
        cmap = kwargs.get("cmap", None)
        alpha = kwargs.get("alpha", 1.0)
        node_size = kwargs.get("node_size", 600)
        nodelist = kwargs.get("nodelist", self.neurons)
        node_color = kwargs.get("node_color", 'r')
        node_borders = kwargs.get("node_borders", "black")
        nx.draw_networkx_nodes(self.network, pos, nodelist=nodelist, alpha=alpha, node_size=node_size, cmap=cmap, node_color=node_color, edgecolors=node_borders)

        # Draw edges
        width = kwargs.get("width", 1.0)
        edge_alpha = kwargs.get("edge_alpha", 1.0)
        edge_color = kwargs.get("edge_color", 'r')
        edge_cmap = kwargs.get("edge_cmap", None)
        edgelist = kwargs.get("edgelist", self.connections)
        nx.draw_networkx_edges(self.network, pos, edgelist=edgelist, alpha=edge_alpha, width=width, edge_color=edge_color, edge_cmap=edge_cmap)

        # Draw labels
        if kwargs.get("labels", True):
            nx.draw_networkx_labels(self.network, pos, font_size=kwargs.get("font_size", 10))

        plt.title(kwargs.get("title", None))
        plt.axis("off")

        if kwargs.get("savefig", False):
            plt.savefig(kwargs.get("title", "my_neuron_network.pdf"), format="pdf", dpi=kwargs.get("dpi", 600))

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

    # TODO: Implement & document!!
    def small_world_propensity(self, weight="weight"):
        """Compute Small-World Propensity (SWP) of the neuron network.

        Source: https://www.nature.com/articles/srep22057

        Args:
            weight: str, optional, default: 'weight'

        Returns:
            small_worldness: float
                The clustering coefficient divided by the average shortest path
                length of the neuron network.
        """
        raise NotImplementedError("Patience is a virtue.")
