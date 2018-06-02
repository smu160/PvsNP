""" This module contains all the functions necessary for
feature engineering, data visualization, and data analysis.
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import seaborn as sns
from scipy import stats

class FeatureExtractor(object):
    """Utilize this class to store, manipulate, and visualize all data pertaining
    to the data garnered from a single experiment
    """

    def __init__(self, cell_transients_df=None, auc_df=None, behavior_df=None, **kwargs):
        self.cell_transients_df = cell_transients_df
        self.auc_df = auc_df

        if isinstance(behavior_df, pd.DataFrame):
            self.behavior_df = behavior_df

            behavior_column_names = kwargs.get("behavior_col_names", None)
            self.behavior_df.columns = behavior_column_names

            row_multiple = kwargs.get("row_multiple", 3)
            self.behavior_df = FeatureExtractor.downsample_dataframe(behavior_df, row_multiple)

            # Adds "Running_frames" column to the end of the behavior Dataframe
            velocity_cutoff = kwargs.get("velocity_cutoff", 4)
            self.behavior_df = self.behavior_df.assign(
                Running_frames=np.where(self.behavior_df["Velocity"] > velocity_cutoff, 1, 0))
            self.neuron_concated_behavior = self.auc_df.join(self.behavior_df, how="left")
        else:
            message = "A behavior dataframe was not provided."
            warnings.warn(message, Warning)
    
    @staticmethod
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
            dataframe: DataFrame

                the downsampled pandas DataFrame
        """
        # Drop every nth (row multiple) row
        dataframe = dataframe.iloc[0::row_multiple, :]

        # Reset and drop the old indices of the pandas DataFrame
        dataframe.reset_index(inplace=True, drop=True)
        return dataframe

    def compute_diff_rate(self, dataframe, neuron_col_names, *behaviors, frame_rate=10):
        """Computes difference between the rates of two behaviors

        Args:
            dataframe: DataFrame

                a concatenated pandas DataFrame of all the neuron column vectors
                for a given animal and its corresponding behavior column
                vectors.

                neuron_col_names: list

                    the names of the neuron column vectors to be computed.

                *behaviors: str

                    a single or ordered pair of behaviors to compute the
                    difference rate for, e.g. "Running", or "ClosedArms",
                    "OpenArms".

                frame_rate: int, optional

                    the framerate associated with the given data; default is 10

        Returns: numpy array

                a numpy array of all the means of the behavior vectors subtracted
                from the corresponding means of the non-behavior vectors, all scaled
                by frame rate
                :param neuron_col_names:
                :param frame_rate:
        """
        if len(behaviors) == 1:
            beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, neuron_col_names]
            no_beh_vec = dataframe.loc[dataframe[behaviors[0]] == 0, neuron_col_names]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
        elif len(behaviors) == 2:
            beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, neuron_col_names]
            no_beh_vec = dataframe.loc[dataframe[behaviors[1]] != 0, neuron_col_names]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
        else:
            raise ValueError("You provided an appropriate amount of behaviors.")

    def set_real_diff_df(self, dataframe, neuron_col_names, beh1, beh2):
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

                a pandas DataFrame that consists of one row with all of the
                actual D_hat values, computed for all the neurons for a given
                animal.
                :param neuron_col_names:
                :param behavior1:
                :param behavior2:
        """
        real_diff_vals = pd.DataFrame(columns=neuron_col_names, index=["D"])
        real_diff_vals.loc['D'] = self.compute_diff_rate(dataframe, neuron_col_names, beh1, beh2)
        return real_diff_vals

    def neuron_line_plot(self, dataframe, *neurons):
        """Plots a line plot of neuron activity over time.

        This is a wrapper function for the plotly library line
        plot functionality. It takes any amount of neurons and
        will plot their time series data over a single line,
        for each individual neuron.

        Args:
            neurons: str

                the name(s) of the column vectors in the dataframe.

            dataframe: DataFrame

                the name of one of the available pandas dataframes to use as the
                source of neuron column vectors to plot.
        """
        data = list()
        for neuron in neurons:
            data.append(go.Scatter(x=list(dataframe[neuron].index), y=dataframe[neuron], name=neuron))

        plotly.offline.iplot(data)

    def neuron_scatter_plot_with_reg(self, neuron1, neuron2, dataframe):
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

    def plot_correlation_heatmap(self, dataframe, **kwargs):
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
        _, _ = plt.subplots(figsize=kwargs.get("figsize", (16, 16)))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(dataframe.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .5})

    def plot_clustermap(self, dataframe, **kwargs):
        """Seaborn clustermap wrapper function

        A wrapper function for seaborn to quickly plot a clustermap using the
        "centroid" method to find clusters.

        Args:
            dataframe: DataFrame

                a Pandas dataframe to be plotted in the clustermap

            figsize: tuple, optional

                the size of the clustermap to be plotted, default is 15
        """
        figsize = kwargs.get("figsize", (15, 15))
        cluster_map = sns.clustermap(dataframe.corr(), center=0, linewidths=.75, figsize=figsize, method="centroid",
                                     cmap="vlag")
        cluster_map.ax_row_dendrogram.set_visible(False)
        cluster_map.ax_col_dendrogram.set_visible(False)

    def new_corr_coeff(self, dataframe, neuron_x, neuron_y):
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

    @staticmethod
    def find_correlated_pairs(dataframe, **kwargs):
        """Find all of the correlated pairs of neurons.

        This function returns a dictionary of all the correlated pairs
        of neurons in a given dataframe of time series data, with an
        individual column vector per neuron.

        Args:
            dataframe: DataFrame

                A pandas dataframe, where the columns are individual neurons,
                and the rows represent neuronal acitivty over time

            corr_coeff: float, optional

                The cutoff for the correlation coefficient to use in order to
                consider a given pair of neurons to be correlated, default is
                0.3.

        Returns:
            corr_pairs_dict: dictionary

                A dictionary of <tuple, correlation value> where the tuple is a
                unique, correlated pair, and the corresponding value is
                correlation coefficient of that tuple.
        """
        corr_pairs_dict = {}
        corr_dataframe = dataframe.corr()
        corr_coeff = kwargs.get("corr_coeff", 0.3)

        for i in corr_dataframe.columns:
            for j in corr_dataframe.index:
                if corr_dataframe.at[i, j] >= corr_coeff and i != j:
                    if (i, j) not in corr_pairs_dict and (j, i) not in corr_pairs_dict:
                        corr_pairs_dict[(i, j)] = corr_dataframe.at[i, j]

        return corr_pairs_dict

    def plot_neurons_as_function_of_beh(dataframe, **kwargs):
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

                the neuron column vector to be plotted along the x-axis.

            neuron_y: string
                the neuron column vector to be plotted along the y-axis.

            behavior: string

                the behavior over time (represented in the form of booleans)

            size_of_plot: int
                the size of the scatter plot. default is 8
        """
        x = kwargs.get("neuron_x", None)
        if x is None:
            raise ValueError("You did not provide the neuron_x column vector!")
        y = kwargs.get("neuron_y", None)
        if y is None:
            raise ValueError("You did not provide the neuron_y column vector!")
        beh = kwargs.get("behavior")
        if beh is None:
            raise ValueError("You did not provide a behavior.")
        size = kwargs.get("size", 8)
        sns.lmplot(x=x, y=y, hue=beh, data=dataframe[[x, y, behavior]], size=size)
