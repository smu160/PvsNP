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

from analysis.analysis_utils import z_score_data

class Deconvoluter:
    """
    Attributes:

        cell_transients: DataFrame
            T x N matrix with s.d. vlaues for all the timepoints of the
            qualified transients (all zeros except for transients)

        cell_data: DataFrame
            T x N matrix with all z-scored data.

        cell_auc_df: DataFrame
            T x N matrix with computed transient area under the curve (AUC)
            values (all zeros except for the AUC value assigned to the peak
            timepoint of each transient).
    """

    def __init__(self, raw_data, **kwargs):
        """
        Args:
            raw_data: DataFrame
                Ca2 transient data in T x N format, where N is the # of neuron
                column vectors, and T is the number of observations (rows),
                all in raw format.

            threshold: int, optional, default: 2
                The minimum aplitude size of ca transient data in s.d.

            baseline: float, optional, default: 0.5
                The standard deviation offset value of Ca transient.

            t_half: float, optional, default: 0.2
                The half life of gcamp type used (s).

            frame_rate: int, optional, default: 10
                The frame rate for the data.
        """

        threshold = kwargs.get("threshold", 2)
        baseline = kwargs.get("baseline", 0.5)
        t_half = kwargs.get("t_half", 0.2)
        frame_rate = kwargs.get("frame_rate", 10)

        # z-score all raw data
        self.cell_data = z_score_data(raw_data)

        # Preallocate dataframes
        self.cell_transients = pd.DataFrame(np.zeros((len(raw_data.index), len(raw_data.columns))))
        self.cell_auc_df = pd.DataFrame(np.zeros((len(raw_data.index), len(raw_data.columns))))

        # Define minimum duration of calcium transient based on gcamp type used

        # Simplified from (-ln(A / Ao) / t_half), [A / Ao] = 0.5 at t half-life,
        # [-ln(A / Ao)] = 0.693
        decay_rate = np.log(2) / t_half

        # Minimum (s) duration for ca transient of minimum
        # specified s.d. amplitude threshold
        minimum_duration = -(math.log(baseline / threshold)) / decay_rate

        # Minimum number of frames the ca transient should last above baseline
        self.minimum_frames = round(minimum_duration * frame_rate)

        self.detect_ca_transients_mossy(threshold, baseline)

    def detect_ca_transients_mossy(self, threshold, baseline):
        """Preprocesses neural signal data obtained via Calcium Imaging

        Args:
            threshold: int
                The minimum aplitude size of ca transient data in s.d.

            baseline: float
                The standard deviation offset value of Ca transient.
        """

        # Identify qualified ca transients and generate outputs
        for column in self.cell_data:

            # Find all timepoints where flourescence greater than threshold
            onset = self.cell_data.loc[self.cell_data[column] > threshold].index

            # Find all timepoints where floursecence greater than the
            # baseline (transient offset)
            offset = self.cell_data.loc[self.cell_data[column] > baseline].index

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
                    peak_index = self.cell_data.loc[start:finish+1, column].values.argmax()
                    transient_vector = np.arange(start, finish+1)

                    # Retrieve "cell" index of that peak value
                    max_amp_index = transient_vector[peak_index]

                    # peak_to_offset_vector = list(range(max_amp_index, finish+1))
                    found = True

                    # If the peak value index from start-stop in offset is also
                    # found in onset vector, then the transient exceeded the
                    # 2SD threshold
                    if ((finish+1) - max_amp_index) >= self.minimum_frames and (max_amp_index in onset):

                        # Retrieve "cell" values for all the timepoints of
                        # that transient.
                        self.cell_transients.loc[start:finish+1, column] = self.cell_data.loc[start:finish+1, column]

                        # Create matrix with all calcium transient peak events
                        # (all 0's except for amplitude value at peak timepoint)
                        # cell_events(maxamp_ind, k) = peak_value

                        # Integrate area under curve of transient from start-stop
                        transient_area = np.trapz(self.cell_data.loc[start:finish+1, column])

                        # Add all the calcium transient Area Under Curve values
                        # (all 0's except for AUC value assigned to peak
                        # timepoint)
                        self.cell_auc_df.loc[max_amp_index, column] = transient_area
