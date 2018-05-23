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

def compute_diff_rate(dataframe, neuron_activity_df, *behaviors, frame_rate=10):
    """Computes difference between the rates of two behaviors

    Args:
        dataframe: DataFrame 
            
            a concatenated pandas DataFrame of an animal's neuron
            activity and corresponding behavior
        
            neuron_activity_df: DataFrame 
            
                the names of the neuron columns in the DataFrame
        
            *behaviors: 
                a single or ordered pair of behaviors to compute the
                difference rate for, e.g. "Running", e.g. "ClosedArms", 
                "OpenArms"
        
            frame_rate:

                the framerate associated with the given data; default is 10

    Returns: numpy array
        
            a numpy array of all the means of the behavior vectors subtracted 
            from the corresponding means of the non-behavior vectors, all scaled
            by frame rate
    """
    if len(behaviors) == 1:
        beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, neuron_activity_df.columns]
        no_beh_vec = dataframe.loc[dataframe[behaviors[0]] == 0, neuron_activity_df.columns]
        return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
    elif len(behaviors) == 2:
        beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, neuron_activity_df.columns]
        no_beh_vec = dataframe.loc[dataframe[behaviors[1]] != 0, neuron_activity_df.columns]
        return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))

def downsample_dataframe(dataframe, row_multiple):
    """Downsample a given pandas DataFrame

    Args:
        dataframe: DataFrame
            the pandas DataFrame to be downsampled
        
        row_multiple: int
        
            the row multiple is the rows to be removed.
            e.g., row_multiple of 3 would remove every 3rd row from the
            provided dataframe

    Returns: 
        the downsampled pandas DataFrame
    """
    # Drop every nth (row multiple) row
    dataframe = dataframe.iloc[0::row_multiple, :]

    # Reset and drop the old indices of the pandas DataFrame
    dataframe.reset_index(inplace=True, drop=True)

    return dataframe

def neuron_scatter_plot_with_reg(neuron1, neuron2, dataframe):
    """What function does...

    Args:

    Returns:
    """

    if pd.isnull(dataframe[neuron1]).all():
        return False

    slope, intercept, r_value, _, _ = stats.linregress(dataframe[neuron1], dataframe[neuron2])
    regression_line = slope * dataframe[neuron1] + intercept

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

def neuron_line_plot(dataframe, *neurons):
    """Plots a line plot of neuron activity over time

    This is a wrapper function for the plotly library line
    plot functionality. It takes any amount of neurons and
    will plot their time series data over a single line,
    for each individual neuron.

    Args:
        dataframe: a pandas DataFrame that contains the neuron(s)
        activity time series datato be plotted as lines
    """
    data = list()
    for neuron in neurons:
        data.append(go.Scatter(x=list(range(0, len(dataframe))), y=dataframe[neuron], name=neuron))

    plotly.offline.iplot(data)

def plot_correlation_heatmap(dataframe, **kwargs):
    """Seaborn correlation heatmap wrapper function

    A wrapper function for seaborn to quickly plot a
    correlation heatmap with a lower triangle, only

    Args:
        dataframe: DataFrame 
            
            a Pandas dataframe to be plotted in the correlation
            heatmap
        
        figsize: tuple, optional
            the size of the heatmap to be plotted, default is 16.
    """

    # Generate a mask for the upper triangle
    mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    _, _ = plt.subplots(figsize=kwargs.get("figsize", (16,16)))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataframe.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def plot_clustermap(dataframe, **kwargs):
    """Seaborn clustermap wrapper function

    A wrapper function for seaborn to quickly plot a clustermap using the
    "centroid" method to find clusters.

    Args:
        dataframe: DataFrame 
            
            a Pandas dataframe to be plotted in the clustermap
        
        figsize: tuple, optional
        
            the size of the clustermap to be plotted, default is 15
    """
    figsize = kwargs.get("figsize", (15,15))
    cluster_map = sns.clustermap(dataframe.corr(), center=0, linewidths=.75, figsize=figsize, method="centroid", cmap="vlag")
    cluster_map.ax_row_dendrogram.set_visible(False)
    cluster_map.ax_col_dendrogram.set_visible(False)

def new_corr_coeff(dataframe, neuron_x, neuron_y):
    """A different way to compute the correlation between 2 neurons

    Formula is as follows:
    $$q=\frac{|\vec{n_1} \wedge \vec{n_2}|}{|\vec{n_1} \vee \vec{n_2}|}$$

    Args:
        dataframe: DataFrame
        
            the pandas dataframe in which the 2 neurons are located
        
        neuron_x: string
        
            the name of the neuron column vector located in the passed-in 
            DataFrame
        
        neuron_y: string 
            
            the name of the neuron column vector located in the passed-in 
            DataFrame
        
    Returns:
        The new and improved correlation value, q
    """
    mag_of_neuron_x_and_neuron_y = len(dataframe[(dataframe[neuron_x] != 0) & (dataframe[neuron_y] != 0)])
    mag_of_neuron_x_or_neuron_y = len(dataframe[(dataframe[neuron_x] != 0) | (dataframe[neuron_y] != 0)])
    return mag_of_neuron_x_and_neuron_y / mag_of_neuron_x_or_neuron_y

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
            if corr_dataframe.at[i, j] >= correlation_coeff and i != j:
                if (i, j) not in corr_pairs_dict and (j, i) not in corr_pairs_dict:
                    corr_pairs_dict[(i, j)] = corr_dataframe.at[i, j]

    return corr_pairs_dict

def plot_neurons_as_function_of_beh(dataframe, neuron_x, neuron_y, behavior, size_of_plot=8):
    """ This function plots two neurons as a function of a third variable (behavior)

    Scatter plots allow one to explore the relationship between a pair of
    variables. Often, however, a more interesting question is “how does the
    relationship between these two variables change as a function of a third
    variable?” The best way to separate out a relationship is to plot both
    levels on the same axes and to use color to distinguish them.
    Source: http://seaborn.pydata.org/tutorial/regression.html

    Args:
        dataframe: DataFrame 
        
            a pandas dataframe that has both, neuron activity and
            corresponding behavior
        
        neuron_x: string 
            
            the neuron to be plotted along the x-axis
        
        neuron_y: string 
            the neuron to be plotted along the y-axis
        
        behavior: string 
            
            the behavior over time (represented in the form of booleans)
        
        size_of_plot: int 
            the size of the scatter plot. default is 8
    """
    sns.lmplot(x=neuron_x, y=neuron_y, hue=behavior, data=dataframe[[neuron_x, neuron_y, behavior]], size=size_of_plot)

def is_neuron_selective(resampled_df, real_diff_vals, neuron, behavior_name, high, low):
    """ Classifies a given neuron as selective or non-selective

    Args:
        bootstrapped_df: DataFrame
            
            all the bootstrapped data for the neurons and a given behavior
        
        real_diff_vals: DataFrame 
                
            a single row with the real D_hat values for each corresponding
            neuron

        neuron: Series 
        
            the particular neuron (column vector) to be classified
        
        behavior_name: string 
            
            the name of the behavior to classify the
        
        high: int
        
            the cutoff for the high percentile
        
        low: int 
            
            the cutoff for the low percentile

    Returns:
        the classification of the neuron; either <behavior>, Non-<behavior>,
        or Non-selective
    """
    if real_diff_vals[neuron]['d'] >= np.percentile(resampled_df[neuron], high):
        return behavior_name
    elif real_diff_vals[neuron]['d'] <= np.percentile(resampled_df[neuron], low):
        return "Non-"+behavior_name
    else:
        return "Non-selective"

def classify_neurons_for_beh(resampled_df, real_diff_vals, behavior_name, high, low):
    """Classifies all neurons for one mouse as either selective or non-selective

    This function runs the is_neuron_selective function on all the given neurons
    in the

    Args:
        resampled_df: DataFrame 
            
            all the resampled (simulated) D_hat values for the neurons,
            and a given behavior
        
        real_diff_vals: DataFrame
            
            a single row with the actual D_hat values.
        
        behavior_name: string
            
            the name of the behavior to classify by.
        
        high: float
            the cutoff for the high percentile.
        
        low: float
            the cutoff for the low percentile.

    Returns:
        neurons_dict: dictionary
            
            a dictionary of neurons as keys and their corresponding 
            classification as values.
    """
    neurons_dict = {}
    for neuron in bootstrapped_df.columns:
        neurons_dict[neuron] = is_neuron_selective(bootstrapped_df, real_d_df, neuron, behavior_name, hi_percentile, lo_percentile)

    return neurons_dict

def set_real_diff_df(dataframe, neuron_sig_df, behavior1, behavior2):
    """Compute the real difference mean values for all neurons

    Args:
        dataframe: DataFrame

            the concatenated pandas DataFrame of the neuron activity
            DataFrame and corresponding behavior DataFrame, for a given animal
        
        neuron_sig_df: DataFrame
            the pandas DataFrame of neuron activity, for a given animal.
        
        behavior: string 
            
            the behavior for which to compute the difference rate.

    Returns:
        real_diff_vals: DataFrame

            a pandas DataFrame that consists of one row with all of the actual
            D_hat values, computed for all the neurons for a given animal.
    """
    real_diff_vals = pd.DataFrame(columns=neuron_sig_df.columns, index=["d"])
    real_diff_vals.loc['d'] = compute_diff_rate(dataframe, neuron_sig_df, behavior1, behavior2)
    return real_diff_vals

