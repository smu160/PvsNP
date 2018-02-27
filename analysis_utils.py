import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import Core

from scipy import stats
from IPython.core.interactiveshell import InteractiveShell

plotly.offline.init_notebook_mode(connected=True);

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
