""" This module contains all the functions necessary for
feature engineering, data visualization, and data analysis.
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from scipy import stats
import SigProc


def bin_dataframe(dataframe, amount_of_bins):
    """
    This function returns a list of equally-sized
    sub-dataframes made from the passed-in dataframe

    Args:

    Returns:
    """
    binned_dataframes = list()
    begin = 0
    multiple = math.floor(len(dataframe.index) / amount_of_bins)
    end = multiple
    for _ in range(0, amount_of_bins):
        binned_dataframes.append(dataframe[begin:end])
        begin = end
        end += multiple

    return binned_dataframes

def neuron_scatter_plot_with_reg(neuron1, neuron2, dataframe):
    """ What function does...

    Args:

    Returns:
    """

    if pd.isnull(dataframe[neuron1]).all():
        return False

    slope, intercept, r_value, _, _ = stats.linregress(dataframe[neuron1], dataframe[neuron2])
    regression_line = slope * dataframe[neuron1] + intercept

    # Only continue to plot the data if the correlation coefficient indicates a
    # moderate - strong correlation
    # if abs(r_value) < 0.3:
    #    return False

    fig = {
        'data': [
            {
                'x': dataframe[neuron1],
                'y': dataframe[neuron2],
                'mode': 'markers'
            },
            {
                'x': dataframe[neuron1],
                'y': regression_line,
                'mode': 'lines'
            }
        ],
        'layout': {
            'xaxis': {'title': neuron1},
            'yaxis': {'title': neuron2}
        }
    }

    return fig, r_value

def neuron_line_plot(neuron1, neuron2, dataframe):
    """ What function does...

    Args:

    Returns:
    """
    trace1 = go.Scatter(
        x=list(range(0, len(dataframe))),
        y=dataframe[neuron1],
        name=neuron1
    )

    trace2 = go.Scatter(
        x=list(range(0, len(dataframe))),
        y=dataframe[neuron2],
        name=neuron2
    )

    data = [trace1, trace2]
    return plotly.offline.iplot(data)

def activity_by_neurons(dataframe, neuron, **behaviors):
    """ What function does...

    Args:

    Returns:
    """
    new_df = dataframe
    for behavior in behaviors:
        new_df = new_df[(new_df[behavior] == behaviors[behavior])]

    return 10 * sum(new_df[neuron]) / len(new_df[behavior])

def load_activities_dataframe(dataframe, dataframe2):
    """ What function does...

    Args:

    Returns:
    """
    activities_dict = {}
    behaviors = {'Arena_centerpoint':1, 'Open1_centerpoint':1,
                 'Open2_centerpoint':1, 'Closed1_centerpoint':1,
                 'Closed2_centerpoint':1, 'OpenArms_centerpoint':1,
                 'ClosedArms_centerpoint':1}

    activities_dataframe = pd.DataFrame(index=dataframe2.columns)

    for behavior in behaviors:
        for neuron in dataframe2:
            activities_dict[neuron] = activity_by_neurons(dataframe, neuron, **{behavior:behaviors[behavior]})

        activities_dataframe[behavior] = pd.Series(activities_dict)

    return activities_dataframe

def plot_correlation_heatmap(dataframe, size=16):
    """ Seaborn correlation heatmap wrapper function

    A wrapper function for seaborn to quickly plot a
    correlation heatmap with a lower triangle, only

    Args:
        dataframe: a Pandas dataframe to be plotted in the correlation
        heatmap
        size: the size of the heatmap to be plotted, default is 16
    """

    # Generate a mask for the upper triangle
    mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    _, _ = plt.subplots(figsize=(size, size))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataframe.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def plot_clustermap(dataframe, size=15):
    """ Seaborn clustermap wrapper function

    A wrapper function for seaborn to quickly plot a
    clustermap using the "centroid" method to find
    clusters

    Args:
        dataframe: a Pandas dataframe to be plotted in the clustermap
        size: the size of the clustermap to be plotted, default is 15
    """
    cluster_map = sns.clustermap(dataframe.corr(), center=0, linewidths=.75, figsize=(size, size), method="centroid", cmap="vlag")
    cluster_map.ax_row_dendrogram.set_visible(False)
    cluster_map.ax_col_dendrogram.set_visible(False)

def new_corr_coeff(dataframe, neuron_x, neuron_y):
    """ A different way to compute the correlation between 2 neurons

    Formula is as follows:
    $$q=\frac{|\vec{n_1} \wedge \vec{n_2}|}{|\vec{n_1} \vee \vec{n_2}|}$$

    Args:
        dataframe: a pandas dataframe in which the 2 neurons are located
        neuron_x: one of the neurons to be correlated with
        neuron_y: the other neuron to be correlated with

    Returns:
        The new and improved correlation value, q
    """
    mag_of_neuron_x_and_neuron_y = len(dataframe[(dataframe[neuron_x] != 0) & (dataframe[neuron_y] != 0)])
    mag_of_neuron_x_or_neuron_y = len(dataframe[(dataframe[neuron_x] != 0) | (dataframe[neuron_y] != 0)])
    return mag_of_neuron_x_and_neuron_y / mag_of_neuron_x_or_neuron_y

def run_epm_analysis(raw_files):
    """ Carry out EPM analysis functions on all available raw datasets

    Args:
        raw_files: a list of csv files to be analyzed
    """
    for raw_file in raw_files:
        data = pd.read_csv(raw_file, header=None)
        _, _, cell_transients_dataframe = SigProc.detect_ca_transients_mossy(data, 2, 0.5, 0.2, 10)
        plot_correlation_heatmap(cell_transients_dataframe)
        plot_clustermap(cell_transients_dataframe)

def find_correlated_pairs(dataframe, correlation_coeff=0.3):
    """
    This function finds a returns a dictionary of all the correlated pairs in a
    given dataframe of the neuron time series data collected in an experiment

    Args:
        dataframe: a pandas dataframe, where the columns are individual neurons,
        and the rows represent neuronal acitivty over time
        correlation_coeff: the cutoff correlation coefficient to use in order to
        consider a given pair of neurons to be correlated. default is 0.3

    Returns:
        a dictionary of <tuple, correlation value> where the tuple is a unique,
        correlated pair, and the corresponding value is correlation coefficient
        of that tuple
    """
    corr_pairs_dict = {}
    corr_dataframe = dataframe.corr()

    for i in corr_dataframe.columns:
        for j in corr_dataframe.index:
            if abs(corr_dataframe.at[i, j]) >= correlation_coeff and i != j:
                if (i, j) not in corr_pairs_dict and (j, i) not in corr_pairs_dict:
                    corr_pairs_dict[(i, j)] = corr_dataframe.at[i, j]

    return corr_pairs_dict

def plot_neurons_as_function_of_beh(dataframe, neuron_x, neuron_y, behavior, size_of_plot=8):
    """ This function plots two neurons as a function of a third variable (behavior)

    Scatter plots allow one to explore the relationship between a pair of variables.
    Often, however, a more interesting question is “how does the relationship between
    these two variables change as a function of a third variable?”
    The best way to separate out a relationship is to plot both levels on the same axes
    and to use color to distinguish them.
    Source: http://seaborn.pydata.org/tutorial/regression.html

    Args:
        dataframe: a pandas dataframe that has both, neuron activity and corresponding behavior
        neuron_x: the neuron to be plotted along the x-axis
        neuron_y: the neuron to be plotted along the y-axis
        behavior: the behavior over time (represented in the form of booleans)
        size_of_plot: the size of the scatter plot. default is 8
    """
    sns.lmplot(x=neuron_x, y=neuron_y, hue=behavior, data=dataframe[[neuron_x, neuron_y, behavior]], size=size_of_plot)

def is_neuron_selective(bootstrapped_df, real_d_df, neuron, behavior_name, hi_percentile, lo_percentile):
    """ Classifies a given neuron as selective or non-selective
    
    Args:
        bootstrapped_df: pandas dataframe that has all the bootstrapped data for 
        the neurons and a given behavior
        real_d_df: a pandas dataframe that has a single row with the real d values
        neuron: the particular neuron to be classified
        behavior_name: the name of the behavior to classify the 
        hi_percentile: the cutoff for the high percentile
        lo_percentile: the cutoff for the low percentile
    
    Returns:
        the classification of the neuron; either <behavior>, Non-<behavior>, or Non-selective 
    """
    # print("{} percentile of {} is {},   d_value of {} is {}".format(hi_percentile, neuron, np.percentile(bootstrapped[neuron], 87), neuron, real_d_df[neuron]['d']))
    if real_d_df[neuron]['d'] >= np.percentile(bootstrapped_df[neuron], hi_percentile):
        return behavior_name
    elif real_d_df[neuron]['d'] <= np.percentile(bootstrapped_df[neuron], lo_percentile):
        return "Non-" + behavior_name
    else: 
        return "Non-selective"
    
def classify_neurons_for_beh(bootstrapped_df, real_d_df, behavior_name, hi_percentile, lo_percentile):
    """ Classifies all neurons for one mouse as either selective or non-selective
    
    This function runs the is_neuron_selective function on all the given neurons
    in the 
    
    Args: 
        bootstrapped_df: pandas dataframe that has all the bootstrapped data for 
        the neurons and a given behavior
        real_d_df: a pandas dataframe that has a single row with the real d values
        behavior_name: the name of the behavior to classify the 
        hi_percentile: the cutoff for the high percentile
        lo_percentile: the cutoff for the low percentile
    
    Returns: 
        neurons_dict: a dictionary of neurons as keys and their corresponding classification as values
    """
    neurons_dict = {}
    for neuron in bootstrapped_df.columns:
        neurons_dict[neuron] = is_neuron_selective(bootstrapped_df, real_d_df, neuron, behavior_name, hi_percentile, lo_percentile)

    return neurons_dict

def shuffle_worker(q, n, neuron_activity_df, mouse_behavior_df, behavior):
    """Homebrewed bootstrapping function for decoding

    Bootstrapping function that allows estimation of the sample distribution
    using cyclical shifting of the index of a pandas dataframe. This function
    is a worker for the shuffle() function

    Args:
        n: the number of random shuffles to be performed on the given data
        neuron_activity_df: the neuron activity dataframe for a given mouse
        mouse_behavior_df: the behavior dataframe for a given mouse 
        (must directly correspond with neuron_activity_df)
        behavior: the behavior to be estimated

    Returns:
        A Pandas DataFrame that contains all the neuron and behavior
        data after all the data has been bootstraped
    """ 
    shifted_beh_df = mouse_behavior_df.copy()
    shuffled_df = pd.DataFrame(columns=neuron_activity_df.columns, index=range(1, n+1))
    
    for row in shuffled_df.itertuples():
        shifted_beh_df.set_index(np.roll(mouse_behavior_df.index, random.randrange(1, len(mouse_behavior_df.index))), inplace=True)
        shifted_df = pd.concat([neuron_activity_df, shifted_beh_df], axis=1)
        shuffled_df.loc[row.Index] = compute_d_rate(shifted_df, neuron_activity_df, behavior)

    q.put(shuffled_df)
    
def shuffle(iterations, neuron_activity_df, mouse_behavior_df, behavior):
    """This function will serve as the bootstrapping function used for decoding
    
    Args:
    
    Returns:
    """
    NUM_OF_ROWS = int(iterations / 10)
    q = Queue()
    processes = []
    rets = []
    for _ in range(0, 10):
        p = Process(target=shuffle_worker, args=(q, NUM_OF_ROWS, neuron_activity_df, mouse_behavior_df, behavior))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    return pd.concat(rets, ignore_index=True)