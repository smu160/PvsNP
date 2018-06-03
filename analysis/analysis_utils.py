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

class FeatureExtractor(object):
    """Utilize this class to store, manipulate, and visualize all data pertaining
    to the data garnered from a single experiment
    """

    def __init__(self, cell_transients_df=None, auc_df=None, behavior_df=None, **kwargs):
        self.cell_transients_df = cell_transients_df
        self.auc_df = auc_df

        if isinstance(behavior_df, pd.DataFrame):
            row_multiple = kwargs.get("row_multiple", None)
            if row_multiple is None:
                warnings.warn("Row multiple to downsample behavior dataframe"
                              + " not specified. Behavior dataframe will be"
                              + " downsampled by a row multiple of 3", Warning)
                row_multiple = 3

            self.behavior_df = FeatureExtractor.downsample_dataframe(behavior_df, row_multiple)
            behavior_column_names = kwargs.get("behavior_col_names", None)
            self.behavior_df.columns = behavior_column_names

            # Adds "Running_frames" column to the end of the behavior Dataframe
            velocity_cutoff = kwargs.get("velocity_cutoff", 4)
            running_frames = np.where(self.behavior_df["Velocity"] > velocity_cutoff, 1, 0)
            self.behavior_df = self.behavior_df.assign(Running_frames=running_frames)
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

    def compute_diff_rate(self, dataframe, col_names, *behaviors, **kwargs):
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

                    The frame rate associated with the given data; default is
                    10.

        Returns: numpy array

                a numpy array of all the means of the behavior vectors
                subtracted from the corresponding means of the non-behavior
                vectors, all scaled by the frame rate.
        """
        frame_rate = kwargs.get("frame_rate", None)
        if frame_rate is None:
            warnings.warn("Frame rate wasn't specified, so frame rate will be"
                          + " set to 10", Warning)
            frame_rate = 10

        if len(behaviors) == 1:
            beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, col_names]
            no_beh_vec = dataframe.loc[dataframe[behaviors[0]] == 0, col_names]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
        elif len(behaviors) == 2:
            beh_vec = dataframe.loc[dataframe[behaviors[0]] != 0, col_names]
            no_beh_vec = dataframe.loc[dataframe[behaviors[1]] != 0, col_names]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
        else:
            raise ValueError("Improper amount of behaviors detected!")

    def set_real_diff_df(self, dataframe, col_names, *behaviors, **kwargs):
        """Compute the real difference mean values for all neurons

        Args:
            dataframe: DataFrame

                The concatenated pandas DataFrame of the neuron activity
                DataFrame and corresponding behavior DataFrame, for a given
                animal.

            col_names:

                A list of the neuron column vector names.

            behaviors: string

                The behaviors for which to compute the difference rate, i.e.,
                D_hat.

        Returns:
            real_diff_vals: DataFrame

                A pandas DataFrame that consists of one row with all of the
                actual D_hat values, computed for all the neurons for a given
                animal.
        """
        frame_rate = kwargs.get("frame_rate", None)
        real_diff_vals = pd.DataFrame(columns=col_names, index=["D"])
        real_diff_vals.loc['D'] = self.compute_diff_rate(dataframe, col_names,
                                                         *behaviors,
                                                         frame_rate=frame_rate)
        return real_diff_vals

    def neuron_line_plot(self, *neurons, **kwargs):
        """Plots a line plot of neuron activity over time.

        This is a wrapper function for the plotly library line plot
        functionality. This function takes any amount of neurons, and plots
        their time series data over a single line, each.

        Args:
            neurons: str

                the name(s) of the column vectors in the dataframe.

            dataframe: str, optional

                The name of one of the available pandas dataframes to use as the
                source of neuron column vectors to plot; default is the
                cell_transients_df. E.g. pass-in dataframe=object.auc_df to use
                the area under the curve dataframe.
        """
        if not neurons:
            raise ValueError("You need to provide the name of at least one"
                             + " neuron column vector in the dataframe.")

        dataframe = kwargs.get("dataframe", None)
        if dataframe is None:
            warnings.warn("You did not specify which dataframe to use for"
                          + " plotting, so the cell transients dataframe"
                          + " was used.", Warning)

            dataframe = self.cell_transients_df

        data = list()
        for neuron in neurons:
            x_axis = list(dataframe[neuron].index)
            y_axis = dataframe[neuron]
            data.append(go.Scatter(x=x_axis, y=y_axis, name=neuron))

        plotly.offline.iplot(data)

    def plot_correlation_heatmap(self, dataframe, **kwargs):
        """Seaborn correlation heatmap wrapper function

        A wrapper function for seaborn to quickly plot a
        correlation heatmap with a lower triangle, only

        Args:
            dataframe: DataFrame

                A Pandas dataframe to be plotted in the correlation heatmap.

            figsize: tuple, optional

                The size of the heatmap to be plotted, default is (16, 16).
        """

        # Generate a mask for the upper triangle
        mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        _, _ = plt.subplots(figsize=kwargs.get("figsize", (16, 16)))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(dataframe.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    def plot_clustermap(self, dataframe, **kwargs):
        """Seaborn clustermap wrapper function

        A wrapper function for seaborn to quickly plot a clustermap using the
        "centroid" method to find clusters.

        Args:
            dataframe: DataFrame

                A Pandas dataframe to be plotted in the clustermap.

            figsize: tuple, optional

                the size of the clustermap to be plotted, default is (15, 15).
        """
        figsize = kwargs.get("figsize", (15, 15))
        cluster_map = sns.clustermap(dataframe.corr(), center=0, linewidths=.75,
                                     figsize=figsize, method="centroid",
                                     cmap="vlag")
        cluster_map.ax_row_dendrogram.set_visible(False)
        cluster_map.ax_col_dendrogram.set_visible(False)

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
