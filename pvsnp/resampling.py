#
# PvsNP: toolbox for reproducible analysis & visualization of neurophysiological data.
# Copyright (C) 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
This module contains all the functions responsible for executing a
permutation test.
"""

__author__ = "Saveliy Yusufov"
__date__ = "1 March 2019"
__license__ = "GPL"
__maintainer__ = "Saveliy Yusufov"
__email__ = "sy2685@columbia.edu"

import os
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import pandas as pd


class Resampler:
    """
    This class is meant to be a toolbox for the purposes of executing
    permutation resampling, and in order to carry out permutation tests.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_num_of_events(dataframe, neuron):
        """Get the number of signal spikes for a given column vector

        Parameters
        ----------
        dataframe: DataFrame

            A pandas DataFrame that contains at least one neuron's signal
            data, in column vector form.

        neuron:
            The name of the neuron column vector to get the number
            of events for.

        Returns
        -------
        The amount of datapoints in a given column vector of nonzero
        value.
        """
        return len(dataframe.loc[:, neuron][dataframe[neuron] != 0])

    @staticmethod
    def diff_of_mean_rate(dataframe, *beh_col_vec, frame_rate=10):
        """Compute difference of means between the rates of two behaviors.

        Parameters
        ----------
        dataframe: DataFrame
            A pandas DataFrame of all the neuron column vectors
            for a given animal.

        beh_col_vec: pandas Series
            A single behavior column vector or two behavior column vectors
            used to compute the difference of means rate.
            e.g. "OpenArms" vs. "ClosedArms".

        frame_rate: int, optional, default: 10
            The framerate associated with the given data.

        Returns
        -------
        numpy array
            A numpy array of all the difference of means, D_hat, values, i.e.,
            all of the behavior vectors means subtracted from the corresponding
            means of the non-behavior vectors, all scaled by the frame rate.
        """
        if len(beh_col_vec) == 1:
            beh_vec = dataframe.loc[beh_col_vec[0] != 0]
            no_beh_vec = dataframe.loc[beh_col_vec[0] == 0]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))
        if len(beh_col_vec) == 2:
            beh_vec = dataframe.loc[beh_col_vec[0] != 0]
            no_beh_vec = dataframe.loc[beh_col_vec[1] != 0]
            return frame_rate * (beh_vec.values.mean(axis=0) - no_beh_vec.values.mean(axis=0))

        raise ValueError("You provided an appropriate amount of behaviors.")

    @staticmethod
    def __shuffle_worker(queue, resamples, dataframe, statistic, *beh_col_vec, **kwargs):
        """Helper function for shuffle()

        This function repeats the permutation resampling and computation of
        the test statistic (difference of means), a *resamples* amount of times.
        This allows us to create a permutation distribution for each neuron
        column vector, under the condition that the null hypothesis is true,
        i.e., that the neuron is not-selective for the behaviors.

        NOTE: This function is meant to be only be used as a helper function
        for the shuffle() function.

        Parameters
        ----------
        queue: Queue
            A thread-safe FIFO data structure to which the resulting
            dataframe will be added.

        resamples: int
            The amount of permutation resamples to draw from the data.

        dataframe: DataFrame
            The data to be used to randomly draw permutation resamples.

        statistic: function
            A function that will compute a statistic that measures the size
            of an effect of interest (e.g. difference of means, mutual
            information, and etc.)

        beh_col_vec: pandas Series
            The columns vectors to be used as the two groups to
            use for permutation resamples.

        flip_roll: boolean, optional, default: False
            If data should be flipped and then randomly rolled for each
            resample.
        """
        flip_roll = kwargs.get("flip_roll", False)

        if isinstance(dataframe, pd.DataFrame):
            column_names = list(dataframe.columns)
        elif isinstance(dataframe, pd.Series):
            column_names = [1]

        rows_list = []
        for _ in range(resamples):
            if flip_roll:
                dataframe = dataframe.reindex(np.roll(dataframe.index, np.random.randint(1, high=len(dataframe.index)+1)))
            else:
                dataframe = dataframe.sample(frac=1)

            dataframe.index = pd.RangeIndex(len(dataframe.index))
            row = statistic(dataframe, *beh_col_vec)

            if len(column_names) > 1:
                rows_list.append(dict(zip(column_names, row)))
            else:
                rows_list.append({column_names[0]: row})

        queue.put(pd.DataFrame(rows_list, columns=column_names))

    @staticmethod
    def shuffle(dataframe, statistic, *beh_col_vec, **kwargs):
        """Permutation resampling function for neuron selectivty analysis.

        This function simply starts a new process for each CPU that the machine
        has. More specifically, this function starts the shuffle_worker()
        function for each new process, in order to allow the permutation
        distribution for each neuron column vector to be created in a more
        expedited fashion. More specifically, the amount of required permutation
        resamples is split evenly amongst all of the CPU's of the machine this
        function will be run on.

        Parameters
        ----------
        resamples: int, optional, default: 10000
            The total amount of permutation resamples desired.

        dataframe: DataFrame
            The data to be used to randomly draw permutation resamples.

        statistic: function
            A function that will compute a statistic that measures the size
            of an effect of interest (e.g. difference of means, mutual
            information, and etc.)

        beh_col_vec: pandas Series
            The columns vectors to be used as the two groups to
            use for permutation resamples.

        flip_roll: boolean, optional, default: False
            If data should be flipped and then randomly rolled for each
            resample.

        Returns
        -------
        pandas DataFrame
            A (vertically) concatenated pandas DataFrame of all the DataFrames
            the shuffle_worker processes produced.
        """
        flip_roll = kwargs.get("flip_roll", False)
        resamples = kwargs.get("resamples", 10000)

        if flip_roll:
            dataframe = dataframe.reindex(np.flip(dataframe.index))
            dataframe.index = pd.RangeIndex(len(dataframe.index))

        keywords = {"flip_roll": flip_roll}
        resamples_per_worker = resamples // os.cpu_count()
        queue = Queue()
        processes = []

        for _ in range(os.cpu_count()):
            process = Process(target=Resampler.__shuffle_worker, args=(queue, resamples_per_worker, dataframe, statistic, *beh_col_vec), kwargs=keywords)
            processes.append(process)
            process.start()

        rets = [queue.get() for process in processes] # queue.get() will block

        for process in processes:
            process.join()

        return pd.concat(rets, ignore_index=True)

    @staticmethod
    def p_value(original_statistic, permutation_distribution):
        """Compute a two-sided p-value.

        This function is meant to compute a two-sided p-value on a given
        permutation distribution.

        Parameters
        ----------
        original_statistic: float
            The original value of the statistic computed on the data.

        permutation_distribution: DataFrame or Series
            A pandas DataFrame of the permutation distributions.

        Returns
        -------
        p_val: float
            The p-value that was located/computed.
        """
        D = original_statistic
        D_i = permutation_distribution.abs()
        P = len(permutation_distribution.index)
        p_val = (1/P) * len(permutation_distribution.loc[D_i >= abs(D)])

        return p_val

    @staticmethod
    def z_score(original_statistic, permutation_distribution):
        """
        Compute z-score for a given value, and the permutation distribution

        Parameters
        ----------
        original_statistic: float
            The original value of the statistic computed on the data.

        permutation_distribution: pandas DataFrame or pandas Series
            The permutation distribution.

        Returns
        -------
        z_score: float
            The z-score that was computed.
        """
        mew = permutation_distribution.mean()
        std = permutation_distribution.std()
        z_score = (original_statistic - mew) / std

        return z_score

    @staticmethod
    def two_tailed_test(original_statistic, permutation_distribution, **kwargs):
        """Conduct a two-tailed hypothesis test.

        WARNING: Use this function ONLY if your permutation distribution is
        normally distributed.

        Parameters
        ----------
        original_statistic: float
            The original value of the statistic computed on the data.

        permutation_distribution: pandas DataFrame
            A DataFrame of the permutation distributions.

        high: float, optional, default: 95.0
            The cutoff for the upper-tail of the distribution.

        low: float, optional, default: 5.0
            The cutoff for the lower-tail of the distribution.

        Returns
        -------
        int
            1 if the original statistic is observed in the upper-tail of the
            permutation distribution.
            -1 if the original statistic is observed in the lower-tail of the
            permutation distribution.
            0 if the original statistic is not observed in either tail of the
            permutation distribution.
        """
        high = kwargs.get("high", 95.0)
        low = kwargs.get("low", 5.0)

        if original_statistic >= np.percentile(permutation_distribution, high):
            return 1
        if original_statistic <= np.percentile(permutation_distribution, low):
            return -1

        return 0
