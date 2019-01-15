"""
This module contains all the functions necessary for data preprocessing as well
as data wrangling.

@authors: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
          Jack Berry, Columbia University, jeb2242@columbia.edu
"""

import os
import sys
import numpy as np
import pandas as pd


def preprocess_behavior(etho_filepath, observer_filepath):
    """
    Processes original ethovision and observer files (converted to .csv format)

    Args:
        etho_filepath: str
            The file path to raw ethovision behavior .csv file (only 1 sheet).

        observer_filepath: str
            The file path to the raw observer .csv file.

    Returns:
        behavior: DataFrame
            The preprocessed behavior DataFrame.
    """

    # read ethovision data and rename columns
    nrowsheader = pd.read_csv(etho_filepath, nrows=0, encoding="latin1", header=0)
    nrowsheader = int(nrowsheader.columns[1])

    behavior = pd.read_csv(etho_filepath, header=nrowsheader-2, encoding="latin1")
    behavior = behavior.drop(behavior.index[0])

    # rename columns
    new_cols = []
    for col in behavior.columns:
        col = col.replace(' ', '_').replace("In_zone(", '').replace("_/_center-point)", '')
        new_cols.append(col)

    behavior.columns = new_cols

    # read observer data and format
    obs = pd.read_csv(observer_filepath)
    obs = obs.drop(["Observation", "Event Log", "Time"], axis=1)
    obs.index += 1
    obs.fillna(0, inplace=True)

    # merge behavior and observer dataframes, and return result
    num_obs = len(obs.columns) # number of hand-scored observations
    obs[obs != 0] = 1
    behavior = pd.merge(behavior, obs, how="left", left_index=True, right_index=True)
    behavior.update(behavior[behavior.columns[-num_obs:]].fillna(0))

    return behavior


def z_score_data(data, mew=None):
    """This function simply z scores all the given neural signal data.

    Args:
        data: DataFrame
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

    z_scored_data = pd.DataFrame(data=((data.values - mew) / sigma))
    return z_scored_data


def pairwise_dist(x_coords):
    """Computes consecutive pairwise differences in a Series.

    NOTE: This is a helper function for distance_moved.

    Input:
        x_coords: pandas Series
            A one-dimensional ndarray of x coordinates, over time.

    Returns:
        dx: pandas Series
            A series containing all the values x[i] - x[i-1]

    """
    x_coords = x_coords.astype(float)
    delta_x = (np.roll(x_coords, -1) - x_coords).shift(1)
    return delta_x


def distance_moved(x_coords, y_coords):
    """
    Args:
        x_coords: pandas Series
            A one-dimensional ndarray of x coordinates, over time.

        y_coords:
            A one-dimensional ndarray of y coordinates, over time.

    Returns
        Series containing distance moved per frame

    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords are not of equal length!")

    delta_x = pairwise_dist(x_coords.astype(float))
    delta_y = pairwise_dist(y_coords.astype(float))

    dist_moved = delta_x**2 + delta_y**2
    dist_moved = dist_moved.apply(np.sqrt)
    return dist_moved.apply(np.sqrt)


def compute_velocity(dist_moved, framerate=10):
    """
    Args:
        dist_moved: pandas Series
            A one-dimensional ndarray containing distance moved.

        framerate: int, optional, default: 10
            The frame rate corresponding to the dist_moved Series.

    Returns:
        velocity: pandas Series
            A one-dimensional ndarray containing velocity.
    """
    velocity = dist_moved.apply(lambda x: x * framerate)
    return velocity


def define_immobility(velocity, min_dur=1, min_vel=2, framerate=10, min_periods=1):
    """Define time periods of immobility based on a rolling window of velocity.

    A Mouse is considered immobile if velocity has not exceeded min_vel for the
    previous min_dur seconds.

    Default values for min_dur and min_vel are taken from:
    Stefanini...Fusi et al. 2018 (https://doi.org/10.1101/292953)

    Args:
        velocity: pandas Series
            A one-dimensional ndarray of the velocity data.

        min_dur: int, optional, default: 1
            The minimum length of time in seconds in which velocity must be low.

        min_vel: int, optional, default: 2
            The minimum velocity in cm/s for which the mouse can be considered
            mobile.

        framerate: int, optional, default: 10
            The frame rate of the velocity Series. Default is 10 fps.

        min_periods: int, optional, default: 1
            Minimum number of datapoints needed to determine immobility. This
            value is needed to define immobile time bins at the beginning of the
            session. If min_periods=8, then the first 8 time bins will be be
            considered immobile, regardless of velocity.

    Returns:
        mobile_immobile: pandas Series
            A one-dimensional ndarray of 0's and 1's, where 1 signifies immobile
            times and 0 signifies mobile times.

    """
    window_size = framerate * min_dur
    rolling_max_vel = velocity.rolling(window_size, min_periods=min_periods).max()
    mobile_immobile = (rolling_max_vel < min_vel).astype(int)

    return mobile_immobile


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
        raise FileNotFoundError("{} does not exist!".format(root_directory))
    if not os.path.isdir(root_directory):
        raise NotADirectoryError("{} is not a directory!".format(root_directory))

    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file == target_file:
                file_path = os.path.join(subdir, file)
                return file_path

    raise FileNotFoundError("{} not found!".format(target_file))


def extract_epochs(mouse, behavior):
    """Extract all epochs of a continuous behavior/event.

    Args:
        mouse: Mouse
            An instance of the Mouse class that has a `spikes_and_beh`
            pandas DataFrame, where `spikes_and_beh` has neural activity
            and corresponding behavior concatenated.

        behavior: str
            The name of the behavior to use when extracting continuous time
            periods/events.

    Returns:
        epochs: pandas Series
            A Series that has hierarchical indices, where "behavior" is the
            outermost index, followed by "interval". Each interval contains
            an ndarray of timepoints in which the corresponding behavior was
            observed, continuously.
    """
    if behavior not in mouse.spikes_and_beh.columns:
        raise ValueError("'{}' is not a column (i.e. behavior) in the mouse's dataframe".format(behavior))

    dataframe = mouse.spikes_and_beh.copy()

    # Find all timepoints where the behavior discontinues
    dataframe["interval"] = (dataframe[behavior].shift(1) != dataframe[behavior]).astype(int).cumsum()

    # Put the index into the dataframe as a column, without creating a new DataFrame
    dataframe.reset_index(inplace=True)

    # Group the dataframe by behavior and corresponding intervals,
    # and apply np.array to the index column of each group.
    epochs = dataframe.groupby([behavior, "interval"])["index"].apply(np.array)

    return epochs


def filter_epochs(interval_series, framerate=10, seconds=1):
    """Helper function for extract_epochs.

    Args:
        interval_series: pandas Series
            A Series with an ndarray of continous behavior timepoints, for each
            index.

        framerate: int, optional, default: 10
            The framerate that corresponds to the session from which the
            intervals were extracted.

        seconds: int, optional, default: 1
            The amount of seconds.

    Returns:
        intervals: list
            A list of all the intervals that are at least as long as the
            provided framerate multiplied by the provided seconds.
    """
    intervals = [interval for interval in interval_series if len(interval) >= framerate*seconds]
    return intervals


class Mouse:
    """A base class for keeping all relevant & corresponding objects, i.e.,
    spikes, cell transients, & behavior dataframes, with their respective
    mouse.
    """

    def __init__(self, cell_transients=None, spikes=None, behavior=None, **kwargs):
        self.name = kwargs.get("name", None)
        self.age = kwargs.get("age", None)
        self.sex = kwargs.get("sex", None)

        self.cell_transients = cell_transients
        self.spikes = spikes

        if behavior is not None:
            self.behavior = behavior
            self.spikes_and_beh = self.spikes.join(self.behavior, how="left")
        else:
            print("A behavior dataframe was not provided.", file=sys.stderr)

        velocity_cutoff = kwargs.get("velocity_cutoff", None)

        # Adds "Running_frames" column to the end of the behavior Dataframe
        if velocity_cutoff is not None:
            running_frames = np.where(self.behavior["Velocity"] > velocity_cutoff, 1, 0)
            self.behavior = self.behavior.assign(Running_frames=running_frames)


def downsample_dataframe(dataframe, row_multiple):
    """Downsample a given pandas DataFrame

    Args:
        dataframe: DataFrame
            The pandas DataFrame to be downsampled.

        row_multiple: int
            The row multiple is the rows to be removed,
            e.g., a row_multiple of 3 would remove every 3rd row from the
            provided dataframe

    Returns:
        dataframe: DataFrame
            The downsampled pandas DataFrame.
    """

    # Drop every nth (row multiple) row
    dataframe = dataframe.iloc[0::row_multiple, :]

    # Reset and drop the old indices of the pandas DataFrame
    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


def activity_by_neurons(spikes_and_beh, neuron_names, *behaviors, **kwargs):
    """Computes the neuron activity rates for given behaviors

    This function computes the rates for a given animal's activity and
    neuron, given some set of behaviors.

    Args:
        spikes_and_beh: DataFrame
            A concatenated pandas DataFrame of the neuron activity and
            the corresponding behavior, for a given animal.

        neuron_names: list
            The names of the neurons whose rates are to be computed.

        behaviors: arbitrary argument list
            The behaviors for which to compute the activity rates.
            NOTE: If no behaviors are provided, then the average rate over the
            entire session.

        frame_rate: int, optional, default: 10
            The framerate by which to multiply the activity rate.

    Returns:
        activity_df: DataFrame
            A pandas DataFrame of the neuron activity rates.
    """
    if set(neuron_names).issubset(spikes_and_beh.columns) is False:
        raise ValueError("neuron_names is NOT a subset of the columns in spikes_and_beh!")

    frame_rate = kwargs.get("frame_rate", None)

    if frame_rate is None:
        print("You did not specify a frame rate! Defaulting to 10", file=sys.stderr)
        frame_rate = 10

    activity_df = pd.DataFrame(columns=behaviors)

    for behavior in behaviors:
        if behavior in spikes_and_beh.columns:
            activity_df.loc[:, behavior] = frame_rate * spikes_and_beh.loc[spikes_and_beh[behavior] != 0, neuron_names].mean()
        else:
            raise ValueError("{} is not a column (i.e. behavior) in spikes_and_beh.".format(behavior))

    # Return average rate over entire session if no behavior(s) is/are provided
    if not behaviors:
        behavior = "entire_session"
        activity_df = pd.DataFrame(columns=[behavior])
        activity_df.loc[:, behavior] = frame_rate * spikes_and_beh.loc[:, neuron_names].mean()

    return activity_df
