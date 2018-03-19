import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import Core

from scipy import stats
from plotly import tools


def bin_dataframe(dataframe, amount_of_bins):
    'This function returns a list of equally-sized sub-dataframes made from the passed-in dataframe'
    binned_dataframes = list()
    begin = 0
    multiple = math.floor(len(dataframe.index) / amount_of_bins)
    end = multiple
    for i in range(0, amount_of_bins):
        binned_dataframes.append(dataframe[begin:end])
        begin = end
        end += multiple
        
    return binned_dataframes

def activity_by_neurons(dataframe, neuron, **behaviors):
    new_df = dataframe
    for behavior in behaviors:
        new_df = new_df[(new_df[behavior] == behaviors[behavior])]

    return 10 * sum(new_df[neuron]) / len(new_df[behavior])

def neuron_scatter_plot_with_reg(neuron1, neuron2, dataframe):
    if pd.isnull(dataframe[neuron1]).all():
        return False
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(dataframe[neuron1], dataframe[neuron2])
    regression_line = slope * dataframe[neuron1] + intercept

    # Only continue to plot the data if the correlation coefficient indicates a moderate - strong correlation
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
    trace1 = go.Scatter(
        x = list(range(0, len(dataframe))),
        y = dataframe[neuron1],
        name = neuron1
    )

    trace2 = go.Scatter(
        x = list(range(0, len(dataframe))),
        y = dataframe[neuron2],
        name = neuron2
    )

    data = [trace1, trace2]
    return plotly.offline.iplot(data)

def load_Activities_DataFrame(dataframe, dataframe2):
    
    activities_dict = {}
    behaviors = {'Arena_centerpoint':1, 'Open1_centerpoint':1, 'Open2_centerpoint':1, 'Closed1_centerpoint':1, 
                 'Closed2_centerpoint':1, 'OpenArms_centerpoint':1, 'ClosedArms_centerpoint':1}

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
    f, ax = plt.subplots(figsize=(size, size))

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
    
    cm = sns.clustermap(dataframe.corr(), center=0, linewidths=.75, figsize=(size, size), method="centroid", cmap="vlag");
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)
        
def q(dataframe, neuron_x, neuron_y):
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

def run_EPM_analysis(raw_files):
    """ Carry out EPM analysis functions on all available raw datasets

    Args:
        raw_files: a list of csv files to be analyzed
    """ 
    
    for raw_file in raw_files:
        data = pd.read_csv(raw_file, header=None)
        z_scored_dataframe, AUC_dataframe, cell_transients_dataframe = Core.detect_ca_transients_mossy(data, 2, 0.5, 0.2, 10)
        plot_correlation_heatmap(cell_transients_dataframe)
        plot_clustermap(cell_transients_dataframe)
        
def find_correlated_pairs(dataframe, correlation_coeff=0.3):
    """ This function finds a returns a dictionary of all the correlated pairs in a 
        given dataframe of the neuron time series data collected in an experiment
    Args: 
        dataframe: a pandas dataframe, where the columns are individual neurons, and 
        the rows represent neuronal acitivty over time
        correlation_coeff: the cutoff correlation coefficient to use in order to consider
        a given pair of neurons to be correlated. default is 0.3
    Returns:
        a dictionary of <tuple, correlation value> where the tuple is a unique correlated 
        pair and the corresponding value is correlation coefficient of that tuple
        
    """
    corr_pairs_dict = {}
    corr_dataframe = dataframe.corr()

    for i in corr_dataframe.columns:
        for j in corr_dataframe.index:
            if abs(corr_dataframe.at[i, j]) >= correlation_coeff and i != j:
                if (i, j) not in corr_pairs_dict and (j, i) not in corr_pairs_dict:
                    corr_pairs_dict[(i, j)] = corr_dataframe.at[i, j]

    return corr_pairs_dict