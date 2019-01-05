"""
This module contains all the functions necessary for data preprocessing.

@authors: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
          Jack Berry, Columbia University, jeb2242@columbia.edu
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

def preprocess_behavior(etho_filepath, observer_filepath):
    """Processes original ethovision and observer files (in .csv format)

    Args:
        etho_filepath:
            file path to raw ethovision behavior .csv file (only 1 sheet)

        observer_filepath:
            file path to raw observer .csv file

    Returns:
        behavior dataframe

    """
    # Read ethovision data and rename columns
    nrowsheader = pd.read_csv(etho_filepath, nrows=0, encoding="latin1", header=0)
    nrowsheader = int(nrowsheader.columns[1])

    behavior = pd.read_csv(etho_filepath, header=nrowsheader-2, encoding="latin1")
    behavior = behavior.drop(behavior.index[0])

    # Rename columns
    new_cols = []
    for col in behavior.columns:
        col = col.replace(' ', '_')
        col = col.replace("In_zone(", "")
        col = col.replace("_/_center-point)", "")
        new_cols.append(col)

    behavior.columns = new_cols

    # Read observer data and format
    obs = pd.read_csv(observer_filepath)
    obs = obs.drop(["Observation", "Event Log", "Time"], axis=1)
    obs.index += 1
    obs.fillna(0, inplace=True)

    # Merge behavior and observer dataframes, and return result
    num_obs = len(obs.columns) # number of hand-scored observations
    obs[obs != 0] = 1
    behavior = pd.merge(behavior, obs, how="left", left_index=True, right_index=True)
    behavior.update(behavior[behavior.columns[-num_obs:]].fillna(0))

    return behavior

def z_score_data(data, mew=None):
    """This function simply z scores all the given neural signal data.

    Args:
        data:
            Ca2 transient data in T x N format, where N is the # of neuron
            column vectors, and T is the number of observations (rows),
            all in raw format.

        mew: int, optional, default: None
            The mean value for the baseline of the data.
            Note: if the raw data comes from CNMF_E, use 0 for the baseline.

    Returns:
        z_scored_data: DataFrame
            The z scored raw cell data in T x N format.
    """
    pop_offset = np.percentile(data.values, 50)
    sigma = np.std(data.values, axis=0)

    if mew is None:
        mew = np.mean(data.values[data.values < pop_offset])
    else:
        mew = mew

    return pd.DataFrame(data=((data.values - mew) / sigma))

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
    root_directory = os.path.join(os.path.expanduser('~'), root_directory)

    if not os.path.exists(root_directory):
        print("{} does not exist!".format(root_directory), file=sys.stderr)
        return
    if not os.path.isdir(root_directory):
        print("{} is not a directory!".format(root_directory), file=sys.stderr)
        return

    for subdir, _, files in os.walk(root_directory):
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
        epochs_df: DataFrame

    """
    dataframe = mouse.spikes_and_beh.copy()
    dataframe["block"] = (dataframe[behavior].shift(1) != dataframe[behavior]).astype(int).cumsum()
    epochs_df = dataframe.reset_index().groupby([behavior, "block"])["index"].apply(np.array)
    return epochs_df

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

class Mouse:
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
