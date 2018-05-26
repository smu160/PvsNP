"""
This module contains all the functions that are needed
for quickly generating and plotting visualizations of
networks.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
import analysis.analysis_utils as au

class neuron_network(object):
    
    def __init__(self, dataframe):
       self.graph = self.create_graph(dataframe)

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
            G: graph
            
                a NetworkX graph of the neuronal network
        """
        G = nx.Graph()
        G.add_nodes_from(dataframe.columns)
        corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

        for key in corr_pairs:
            G.add_edge(key[0], key[1], weight=round(corr_pairs[key], 3))

        return G

    def create_random_graph(dataframe):
        """Generates a random NetworkX graph

        Each individual column of the provided DataFrame will be represented by
        a single node in the graph. The amount of correlated nodes (neurons) in
        the provided DataFrame will be computed, and that specific amount of
        edges will be added between random pairs of nodes (neurons) in the graph.

        Args:
            dataframe: DataFrame 
                
                the pandas DataFrame to use as a basis for the random graph
        
        Returns: 
            G: graph
                a NetworkX graph of the neuronal network
        """
        G = nx.Graph()
        G.add_nodes_from(dataframe.columns)
        corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

        # Connect a len(correlated_pairs_dict) amount of random edges between 
        # all nodes in the random graph
        end_node1 = np.random.randint(1, len(dataframe.columns)+1)
        end_node2 = np.random.randint(1, len(dataframe.columns)+1) 
        for i in range(len(corr_pairs)):
            G.add_edge(end_node1, end_node2)

        return G

    def plot_network(self, **kwargs):
        """A wrapper function for plotting a NetworkX graph

        This function will draw a provided NetworkX graph using the spring
        layout algorithm.

        Args:
            G: graph
                
                the NetworkX graph to be plotted
        """

        # positions for all nodes
        pos = nx.spring_layout(self.graph, weight="weight")

        # Size of the plot
        plt.figure(figsize=kwargs.get("figsize", (35,35)))

        # nodes
        node_size = kwargs.get("node_size", 1000)
        color = kwargs.get("node_color", "pink")
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color=color);

        edges, weights = zip(*nx.get_edge_attributes(self.graph, "weight").items())

        # edges
        nx.draw_networkx_edges(self.graph, pos, width=3.0, edge_color=weights, edge_cmap=plt.cm.YlGnBu);

        labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        # labels
        font_size = kwargs.get("font_size", 15)
        nx.draw_networkx_labels(self.graph, pos, font_size=font_size, edge_labels=labels)

        plt.axis("off");

        save_to_file = kwargs.get("save", False)
        if save_to_file:
            title = kwargs.get("file_name", "Graph.png")
            file_format = kwargs.get("format", "PNG")
            plt.savefig(title, format=file_format)

        plt.show();

    def generate_random_graph(self):

        # positions for all nodes
        pos = nx.spring_layout(random_graph, weight='weight')

        plt.figure(figsize=(15,15))

        # nodes
        nx.draw_networkx_nodes(random_graph, pos, node_size=700, node_color='lightblue');

        # edges
        nx.draw_networkx_edges(random_graph, pos, width=1.0);

        labels = nx.get_edge_attributes(random_graph, 'weight')
        nx.draw_networkx_edge_labels(random_graph, pos, edge_labels=labels)

        # labels
        nx.draw_networkx_labels(random_graph, pos, font_size=15, edge_labels=labels)

        plt.axis('off');
        plt.show();

    def compute_connection_density(self):
        n = len(list(self.graph.nodes()))
        return len(list(self.graph.edges())) / ((n * (n-1)) / 2)

    def compute_mean_betw_cent(self):
        graph_centrality = nx.betweenness_centrality(self.graph, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_mean_degree_cent(self):
        graph_centrality = nx.degree_centrality(self.graph)
        return np.mean(list(graph_centrality.values()))

    def compute_mean_eigen_cent(self):
        graph_centrality = nx.eigenvector_centrality(self.graph, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_mean_katz_cent(self):
        graph_centrality = nx.katz_centrality(self.graph)
        return np.mean(list(graph_centrality.values()))

    def compute_mean_load_cent(self):
        graph_centrality = nx.load_centrality(self.graph, weight="weight")
        return np.mean(list(graph_centrality.values()))

    def compute_max_clique_size(self):
        "https://en.wikipedia.org/wiki/Clique_(graph_theory)#Definitions"
        return len(clique.max_clique(self.graph))

    def compute_mean_clique_size(self):
        """Computes the mean clique size of a given graph

            Args:
                G: a NetworkX graph

            Returns:
                mean: the mean clique size of the given NetworkX graph, G
        """
        all_cliques = nx.enumerate_all_cliques(self.graph)

        size = 0
        running_sum = 0
        for l in all_cliques:
            size += 1
            running_sum += len(l)

        mean = running_sum / size
        return mean
