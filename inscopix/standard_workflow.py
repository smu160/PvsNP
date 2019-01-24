# -*- coding: utf-8 -*-
"""

"""

import os
import sys

import pandas as pd
import isx


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


def preprocess(file_path, output_directory):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Wrong file or file path")

    rec_files = [file_path]

    # Preprocess the recordings by spatially downsampling by a factor of 2.
    pp_files = isx.make_output_file_paths(rec_files, output_directory, "PP")
    isx.preprocess(rec_files, pp_files, spatial_downsample_factor=2)

    # Perform spatial bandpass filtering with defaults.
    bp_files = isx.make_output_file_paths(pp_files, output_directory, "BP")
    isx.spatial_filter(pp_files, bp_files, low_cutoff=0.005, high_cutoff=0.500)


def main():
    if len(sys.argv) != 3:
        print("Usage: {} file_path output_directory".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1] # "H:/Mossy Cell/Drd87/nVista/Drd87_Day4/recording_20171209_105147.xml"
    if os.path.isfile(file_path):
        print("{} is a file".format(file_path))
    else:
        print("{} is not a file!".format(file_path))
        sys.exit(1)

    output_directory = sys.argv[2]
    if os.path.isdir(output_directory):
        print("{} is a directory".format(output_directory))
    else:
        print("{} is not a directory!".format(output_directory))
        sys.exit(1)

    preprocess(file_path, output_directory)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
