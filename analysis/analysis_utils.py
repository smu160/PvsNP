"""
This module contains all the functions necessary for
feature engineering, data visualization, and data analysis.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def find_file(root_directory, target_file):
    """Finds a file in a given root directory (folder).

    Args:
        root_directory: str
            The name of the first or top-most directory (folder)
            to search for the target file (e.g. "Hen_Lab/Mice").

        target_file: str
            The full name of the file to be found (e.g. "mouse1_spikes.csv").

    Returns:
        file_path: str
            The full path to the target file.

    """
    root_directory = os.path.join(os.path.expanduser("~"), root_directory)

    if not os.path.exists(root_directory):
        print("{} does not exist!".format(root_directory), file=sys.stderr)
        return
    if not os.path.isdir(root_directory):
        print("{} is not a directory!".format(root_directory), file=sys.stderr)
        return

    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file == target_file:
                file_path = os.path.join(subdir, file)
                return file_path

    print("{} not found!".format(target_file), file=sys.stderr)

class Mouse(object):
    """A base class for keeping all relevant & corresponding objects, i.e.,
    spikes, cell transients, & behavior dataframes, with their respective
    mouse.
    """

    def __init__(self, cell_transients=None, spikes=None, behavior=None, **kwargs):
        self.cell_transients = cell_transients
        self.spikes = spikes

        if behavior is not None:
            self.behavior = behavior
            self.spikes_and_beh = self.spikes.join(self.behavior, how="left")
        else:
            message = "A behavior dataframe was not provided."
            warnings.warn(message, Warning)

        # Adds "Running_frames" column to the end of the behavior Dataframe
        velocity_cutoff = kwargs.get("velocity_cutoff", None)
        if velocity_cutoff is not None:
            running_frames = np.where(self.behavior["Velocity"] > velocity_cutoff, 1, 0)
            self.behavior = self.behavior.assign(Running_frames=running_frames)

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
    def activity_by_neurons(concated_df, neuron_names, *behaviors, **kwargs):
        """Computes the neuron activity rates for given behaviors

        This function computes the rates for a given animal's activity and
        neuron, given some set of behaviors.

        Args:
            concated_df: DataFrame
                A concatenated pandas DataFrame of the neuron activity and
                the corresponding behavior, for a given animal.

            neuron_names: list
                The names of the neurons whose rates are to be computed.

            behaviors:
                The behaviors for which to compute the activity rates.

            frame_rate: int, optional
                The framerate to multiply the activity rate by, default is 10.

        Returns:
            activity_df: a pandas DataFrame of the neuron activity rates.
        """
        frame_rate = kwargs.get("frame_rate", None)
        if frame_rate is None:
            warnings.warn("You did not specify a frame rate, so a frame rate"
                          + " of 10 will be utilized in the computation", Warning)
            frame_rate = 10

        activity_df = pd.DataFrame(columns=behaviors)

        for behavior in behaviors:
            if behavior in concated_df.columns:
                activity_df.loc[:, behavior] = frame_rate * concated_df.loc[concated_df[behavior] != 0, neuron_names].mean()
            elif '&' in behavior:
                beh1 = behavior.split('&')[0]
                beh2 = behavior.split('&')[1]
                activity_df.loc[:, behavior] = frame_rate * concated_df.loc[(concated_df[beh1] != 0) & ((concated_df[beh2] != 0)), neuron_names].mean()
            elif '|' in behavior:
                beh1 = behavior.split('|')[0]
                beh2 = behavior.split('|')[1]
                activity_df.loc[:, behavior] = frame_rate * concated_df.loc[(concated_df[beh1] != 0) | ((concated_df[beh2] != 0)), neuron_names].mean()

        return activity_df
