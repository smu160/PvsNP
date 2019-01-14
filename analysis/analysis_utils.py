"""
This module contains all the functions necessary for
feature engineering, data visualization, and data analysis.

@authors: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
          Jack Berry, Columbia University, jeb2242@columbia.edu
"""
import os
import sys
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def preprocess_behavior(etho_filepath, observer_filepath):
    """
    Processes original ethovision and observer files (converted to .csv format)
    
    Args:
        etho_filepath: file path to raw ethovision behavior .csv file (only 1 sheet)
        observer_filepath: file path to raw observer .csv file
    
    Returns: 
        behavior dataframe 
    
    """
    #read ethovision data and rename columns
    nrowsheader = pd.read_csv(etho_filepath, nrows=0, encoding = 'latin1',header=0)
    nrowsheader=int(nrowsheader.columns[1])

    behavior=pd.read_csv(etho_filepath, header=nrowsheader-2, encoding = 'latin1')
    behavior=behavior.drop(behavior.index[0])

    #rename columns
    new_cols=[]
    for s in behavior.columns:
        s=s.replace(" ","_")
        s=s.replace("In_zone(","")
        s=s.replace("_/_center-point)","")
        new_cols.append(s)
    behavior.columns=new_cols
    
    #read observer data and format
    
    obs = pd.read_csv(observer_filepath)
    obs=obs.drop(['Observation','Event Log',"Time"],axis=1)
    obs.index=obs.index+1
    obs.fillna(0,inplace=True)
    
    #merge behavior and observer dataframes, and return result
    num_obs = len(obs.columns) #number of hand-scored observations
    obs[obs != 0] = 1
    behavior=pd.merge(behavior,obs,how='left',left_index=True,right_index=True)
    behavior.update(behavior[behavior.columns[-num_obs:]].fillna(0))
    
    return behavior

def z_score_data(data):
    """This function simply z scores all the given neural signal data.

    Args:
        data:

            Ca2 transient data in T x N format, where N is the # of neuron
            column vectors, and T is the number of observations (rows),
            all in raw format.

    Returns:
        z_scored_data: DataFrame

            The z scored raw cell data in T x N format.
    """
    pop_offset = np.percentile(data.values, 50)
    sigma = np.std(data.values, axis=0)
    #mew = np.mean(data.values[data.values < pop_offset])
    mew = 0 #if raw data comes from CNMFE, use 0 for baseline
    return pd.DataFrame(data=((data.values - mew) / sigma))

def pairwise_dist(x):
    """
    Helper function for distance_moved. Computes consecutive pairwise differences in a Series
    
    Input:
        x: Series containing floats
        
    Returns:
        dx: Series containing x[i] - x[i-1]
    
    """
    
    x = x.astype(float)
    dx = (np.roll(x,-1) - x).shift(1)
    return dx

def distance_moved(x,y):
    """
    Inputs
        x,y: Series containing x and y positions over time
    
    Returns
        Series containing distance moved per frame
    
    """  
    
    dx = pairwise_dist(x.astype(float))
    dy = pairwise_dist(y.astype(float))
    
    dist_moved = dx**2 + dy**2
    
    return dist_moved.apply(math.sqrt)

def compute_velocity(dist,fr=10):
    """
    Inputs
        dist: Series containing distance moved
        fr: frame rate
    Returns
        Series containing velocity 
    
    """
    
    return dist.apply(lambda x: x*fr)

def define_immobility(velocity, min_dur = 1, min_vel = 2, fr = 10, min_periods = 1):
    """
    Defines time periods of immobility based on a rolling window of velocity.
    Mouse is considered immobile if velocity has not exceeded min_vel for the previous min_dur seconds
    Defaults for min_dur and min_vel are taken from Stefanini...Fusi et al. 2018
    
    Inputs:
        velocity: Series with velocity data
        min_dur: minimum length of time in seconds in which velocity must be low. Default is 1 second.
        min_vel: minimum velocity in cm/s for which the mouse can be considered mobile. Default is 2 cm/s
        fr: frame rate of velocity Series. Default is 10 fps.
        min_periods: Minimum number of datapoints needed to determine immobility. Default is 1.
            This value is needed to define immobile time bins at the beginning of the session. If 
            min_periods = 8, then the first 8 time bins will be be considered immobile, regardless of velocity
    
    Returns:
        Series defining immobile (1) and mobile (0) times
    
    """
    
    window_size = fr*min_dur
    rolling_max_vel = velocity.rolling(window_size,min_periods=1).max()

    return (rolling_max_vel<min_vel).astype(int)

    



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

def extract_epochs(mouse, behavior):
    """Extract all epochs of continuous behaviors/events.

    Args:
        mouse: Mouse

        behavior: str

    Returns:
        df: DataFrame

    """
    dataframe = mouse.spikes_and_beh.copy()
    dataframe["block"] = (dataframe[behavior].shift(1) != dataframe[behavior]).astype(int).cumsum()
    df = dataframe.reset_index().groupby([behavior, "block"])["index"].apply(np.array)
    return df

def filter_epochs(interval_series, framerate=10, seconds=1):
    """Helper function for extract_epochs.

    Args:
        interval_series: list

        framerate: int, optional

        seconds: int, optional

    Returns:
        intervals: list

    """

    intervals = []

    for interval in interval_series:
        if len(interval) >= framerate*seconds:
            intervals.append(interval)

    return intervals

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
            #else:
                #raise ValueError("{} is not a column in the provided dataframe.".format(behavior))
        return activity_df

