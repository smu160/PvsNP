import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
import analysis.analysis_utils as au


def create_graph(dataframe):
    G = nx.Graph()
    G.add_nodes_from(dataframe.columns)
    corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

    for key in corr_pairs:
        G.add_edge(key[0], key[1], weight=round(corr_pairs[key], 3))
        
    return G

def create_random_graph(dataframe):
    G = nx.Graph()
    G.add_nodes_from(dataframe.columns)
    corr_pairs = au.find_correlated_pairs(dataframe, correlation_coeff=0.3)

    # Connect a len(correlated_pairs_dict) amount of random edges between all the nodes in the random graph
    for i in range(len(corr_pairs)):
        G.add_edge(np.random.randint(1, len(dataframe.columns)+1), np.random.randint(1, len(dataframe.columns)+1))
        
    return G

def plot_graph(G):

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

def compute_mean_betweenness_centrality(graph):
    graph_centrality = nx.betweenness_centrality(graph)
    return np.mean(list(graph_centrality.values()))