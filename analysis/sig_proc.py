#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:15:17 2018

Ca transient event detection, adapted from Dombeck 2007

This is based on the MATLAB implementation of detect_ca_transients_mossy which
was written by Jessica Jimenez, Columbia University, jcj2123@columbia.edu

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""
import math
import numpy as np
import pandas as pd

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
    mew = np.mean(data.values[data.values < pop_offset])
    return pd.DataFrame(data=((data.values - mew) / sigma))

def detect_ca_transients_mossy(data, thresh, baseline, t_half, frame_rate):
    """Preprocesses neural signal data obtained via Calcium Imaging

    Args:
        data:

            Ca2 transient data in T x N format, where N is the # of neuron
            column vectors, and T is the number of observations (rows),
            all in raw format.

        thresh: int, optional

            The minimum aplitude size of ca transient data in s.d.

        baseline: float, optional

            The standard deviation offset value of Ca transient.

        t_half: float, optional

            The half life of gcamp type used (s).

        frame_rate: int, optional

            The frame rate for the data.

    Returns:
        cell_transients: DataFrame

            T x N matrix with s.d. vlaues for all the timepoints of the
            qualified transients (all zeros except for transients)

        cell_events: DataFrame

            T x N matrix with calcium transient peak events (all 0's except for
            the amplitude value at the peak timepoint of each transient)

        cell_auc_df: DataFrame

            T x N matrix with computed transient area under the curve (AUC)
            values (all zeros except for the AUC value assigned to the peak
            timepoint of each transient).

    """

    # zscore all data
    cell_data = z_score_data(data)

    # Preallocate outputs
    # cell_events = zeros(cell_data_size)
    cell_transients = pd.DataFrame(np.zeros((len(data.index), len(data.columns))))
    cell_auc_df = pd.DataFrame(np.zeros((len(data.index), len(data.columns))))

    # Define minimum duration of calcium transient based on gcamp type used

    # Simplified from (-ln(A / Ao) / t_half), [A / Ao] = 0.5 at t half-life,
    # [-ln(A / Ao)] = 0.693
    decayrate = np.log(2) / t_half

    # Minimum (s) duration for ca transient of minimum
    # specified s.d. amplitude threshold
    minimum_duration = -(math.log(baseline / thresh)) / decayrate

    # Minimum number of frames the ca transient should last above baseline
    minimum_frames = round(minimum_duration * frame_rate)

    # Identify qualified ca transients and generate outputs
    for column in data:

        # Find all timepoints where flourescence greater than threshold
        onset = cell_data.loc[cell_data[column] > thresh].index

        # Find all timepoints where floursecence greater than the
        # baseline (transient offset)
        offset = cell_data.loc[cell_data[column] > baseline].index

        found = True

        for i in range(0, len(offset)-1):
            if found:

                # Specify start index of transient from offset vector
                start = offset[i]

                found = False

            # Specify stop index of transient from offset vector
            if offset[i+1] - offset[i] > 1 or i == len(offset)-2:

                # Deals with the last index in the vector
                if i == len(offset)-2:
                    finish = offset[i+1]
                else:
                    finish = offset[i]

                # Find the peak value in that start-stop range
                # peak_value = cell_data.loc[start:finish+1, column].max()
                peak_index = cell_data.loc[start:finish+1, column].values.argmax()
                transient_vector = np.arange(start, finish+1)

                # Retrieve "cell" index of that peak value
                max_amp_index = transient_vector[peak_index]

                # peak_to_offset_vector = list(range(max_amp_index, finish+1))
                found = True

                # If the peak value index from start-stop in offset is also
                # found in onset vector, then the transient exceeded the
                # 2SD threshold
                if ((finish+1) - max_amp_index) >= minimum_frames and (max_amp_index in onset):

                    # Retrieve "cell" values for all the timepoints of
                    # that transient.
                    cell_transients.loc[start:finish+1, column] = cell_data.loc[start:finish+1, column]

                    # Create matrix with all calcium transient peak events
                    # (all 0's except for amplitude value at peak timepoint)
                    # cell_events(maxamp_ind, k) = peak_value

                    # Integrate area under curve of transient from start-stop
                    transient_area = np.trapz(cell_data.loc[start:finish+1, column])

                    # Create a matrix with all the calcium transient AOC values
                    # (all 0's except for AUC value assigned to peak timepoint)
                    cell_auc_df.loc[max_amp_index, column] = transient_area

    cell_data.columns = ["neuron" + str(i) for i in range(1, len(cell_data.columns)+1)]
    cell_auc_df.columns = ["neuron" + str(i) for i in range(1, len(cell_auc_df.columns)+1)]
    cell_transients.columns = ["neuron" + str(i) for i in range(1, len(cell_transients.columns)+1)]

    return cell_data, cell_auc_df, cell_transients
