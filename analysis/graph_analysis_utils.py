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

def create_graph(dataframe):
    """Wrapper function for creating a NetworkX graph
    
    Each individual column of the provided DataFrame will be represented by
    a single node in the graph. Each pair of correlated nodes (neurons)
    will be connected by an edge, where the edge will receive a weight of the
    specific correlation coefficient of those two nodes.
    
    Args: 
        dataframe: a pandas DataFrame that contains the data to be represented 
        with a NetworkX graph

    Returns:
        G: a NetworkX graph of the neuronal network 
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
        dataframe: the pandas DataFrame to use as a basis for the random graph
    """
    G = nx.Graph()
    G.add_nodes_from(dataframe.columns)
    corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

    # Connect a len(correlated_pairs_dict) amount of random edges between all 
    # nodes in the random graph
    for i in range(len(corr_pairs)):
        G.add_edge(np.random.randint(1, len(dataframe.columns)+1), np.random.randint(1, len(dataframe.columns)+1))
        
    return G

def plot_graph(G):
    """A wrapper function for plotting a NetworkX graph
    
    This function will draw a provided NetworkX graph using the spring_layout
    algorithm. The 
    
    Args:
        G: the NetworkX graph to be plotted
    """

    # positions for all nodes
    pos = nx.spring_layout(G, weight='weight') 

    plt.figure(figsize=(35, 35))

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue');

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    # edges
    nx.draw_networkx_edges(G, pos, width=3.0, edge_color=weights, edge_cmap=plt.cm.YlGnBu);

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=15, edge_labels=labels)

    plt.axis('off');
    plt.show();

def plot_random_graph(random_graph):
    
    # positions for all nodes
    pos = nx.spring_layout(random_graph, weight='weight') 

    plt.figure(figsize=(15, 15))

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
    
def compute_network_measures(graph):
    """
    
    args:
    
    returns:
    """
    network_measures_dict = dict()
    network_measures_dict["assortativity"] = nx.degree_assortativity_coefficient(graph) 
    network_measures_dict["mean betweenness centrality"] = compute_mean_betweenness_centrality(graph)
    #network_measures_dict["mean clique size"] = 
    network_measures_dict["max clique size"] = len(clique.max_clique(graph))
    network_measures_dict["clustering coefficient"] = nx.clustering(graph)
    #network_measures_dict["mean path length"] = 
    
    return network_measures_dict

def compute_connection_density(graph):
    n = len(list(graph.nodes()))
    return len(list(graph.edges())) / ((n * (n-1)) / 2)

def compute_mean_betweenness_centrality(graph):
    graph_centrality = nx.betweenness_centrality(graph, weight="weight")
    return np.mean(list(graph_centrality.values()))

def compute_mean_degree_centrality(graph):
    graph_centrality = nx.degree_centrality(graph)
    return np.mean(list(graph_centrality.values()))

def compute_mean_eigen_centrality(graph):
    graph_centrality = nx.eigenvector_centrality(graph, weight="weight")
    return np.mean(list(graph_centrality.values()))

def compute_mean_katz_centrality(graph):
    graph_centrality = nx.katz_centrality(graph)
    return np.mean(list(graph_centrality.values()))

def compute_mean_load_centrality(graph):
    graph_centrality = nx.load_centrality(graph, weight="weight")
    return np.mean(list(graph_centrality.values()))

def get_max_clique_size(graph):
    "https://en.wikipedia.org/wiki/Clique_(graph_theory)#Definitions"
    return len(clique.max_clique(graph))

def compute_mean_clique_size(graph):
    """Computes the mean clique size of a given graph
    
        Args:
            G: a NetworkX graph
        
        Returns:
            mean: the mean clique size of the given NetworkX graph, G
    """
    all_cliques = nx.enumerate_all_cliques(graph)
    
    size = 0
    running_sum = 0
    for l in all_cliques:
        size += 1
        running_sum += len(l)

    mean = running_sum / size
    return mean
