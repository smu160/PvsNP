"""
This module contains all the functions responsible for executing a
permutation test.
"""

import os
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import pandas as pd
from analysis import analysis_utils as au

class Resampler(object):

    def __init__(self):
        pass

    @staticmethod
    def compute_two_side_p_val(resampled_df, real_diff_vals, neuron):
        """Compute a two-sided p-value for permutation test

            IMPORTANT: Use this function in place when the data you resampled 
            does not is not normally distributed.

            Args:
                resampled_df: DataFrame

                    A pandas DataFrame of all the rates computed after 
                    resampling.

                real_diff_vals: DataFrame

                    A Pandas DataFrame with a single row of all the ACTUAL rate
                    values that were computed for a given set of neurons.

                neuron:

                    The neuron column vector to compute the two-sided p
                    value for.

            Returns:

                The two-sided p-value for a given neuron column vector.
        """
        p = len(resampled_df.index)
        D = real_diff_vals.at['D', neuron]
        D_i = resampled_df[neuron].abs()
        return (1/p) * len(resampled_df.loc[:, neuron].loc[D_i >= abs(D)])

    @staticmethod
    def get_num_of_events(dataframe, neuron):
        """Get the number of signal spikes for a given column vector

           Args:
               dataframe: a Pandas DataFrame that contains at least one
               neuron's signal data, in column vector form.
               neuron: the name of the neuron column vector to get the
               number of events for.

           Returns:
               the amount of datapoints in a given column vector with
               of nonzero value.
        """
        return len(dataframe.loc[:, neuron][dataframe[neuron] != 0])

    @staticmethod
    def compute_diff_rate(dataframe, neuron_col_names, *behaviors, frame_rate=10):
        """Computes difference between the rates of two behaviors

        Args:
            dataframe: DataFrame

                a concatenated pandas DataFrame of all the neuron column vectors for
                a given animal and its corresponding behavior column vectors.

                neuron_col_names: list

                    the names of the neuron column vectors to be computed.

                *behaviors: str

                    a single or ordered pair of behaviors to compute the
                    difference rate for, e.g. "Running", or "ClosedArms",
                    "OpenArms".

                frame_rate: int

                    the framerate associated with the given data; default is 10

        Returns: numpy array

                a numpy array of all the means of the behavior vectors subtracted
                from the corresponding means of the non-behavior vectors, all scaled
                by frame rate
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

    @staticmethod
    def shuffle(total_experiments, neuron_concated_behavior, neuron_col_names, beh1, beh2):
        """Homebrewed resampling function for EPM Analysis

        Resampling function that gives the capability to "simulate"
        experiments using random shuffling of the observations for each
        pandas dataframe.

        Args:
            total_experiments: the total amount of epxeriments to simulate via
            resampling.
            neuron_and_behavior_df: the concatenated neuron activity and behavior
            dataframes
            for a given animal
            neuron_activity_df: the neuron activity dataframe for a given animal
            behavior: the specific behavior to simulate the experiments on

        Returns: a (vertically) concatenated Pandas DataFrame of all the DataFrames
        the shuffle_worker processes produced.
        """
        experiments_per_worker = total_experiments // os.cpu_count()
        queue = Queue()
        processes = []
        rets = []
        for _ in range(os.cpu_count()):
            process = Process(target=Resampler.shuffle_worker, args=(queue, experiments_per_worker, neuron_col_names, neuron_concated_behavior, beh1, beh2))
            processes.append(process)
            process.start()
        for process in processes:
            ret = queue.get()  # will block
            rets.append(ret)
        for process in processes:
            process.join()

        return pd.concat(rets, ignore_index=True)

    @staticmethod
    def shuffle_worker(queue, num_of_experiments, neuron_col_names, neuron_and_behavior_df, beh1, beh2):
        """Helper function for shuffle()

        Given a certain number of experiments to simulate, this function will
        add a dataframe to a provided queue full of the amount of experiments
        desired as obervations rows.
        Note: This function is meant to be only be used as a helper function
        for the shuffle() function

        Args:
            q: the blocking queue to which the resulting dataframe will be added to
            num_of_experiments: the number of experiments that will be simulated
            and appended, as observations, to the dataframe to be returned
            neuron_activity_df: the neuron activity dataframe for a given mouse
            neuron_and_behavior_df: the concatenated neuron activity and behavior
            dataframes for a given mouse
            behavior: the specific behavior to simulate the experiments on
        """
        first_col = neuron_col_names[0]
        last_col = neuron_col_names[len(neuron_col_names)-1]
        shuffled_df = pd.DataFrame(columns=neuron_col_names)

        for index in range(num_of_experiments):
            neuron_and_behavior_df.loc[:, first_col:last_col] = neuron_and_behavior_df.loc[:, first_col:last_col].sample(frac=1).reset_index(drop=True)
            shuffled_df.loc[index] = Resampler.compute_diff_rate(neuron_and_behavior_df, neuron_col_names, beh1, beh2)

        queue.put(shuffled_df)

    @staticmethod
    def non_normal_neuron_classifier(dataframe, resampled_df, real_diff_df, p_value=0.05, threshold=10):
        """Classify neurons as selective or not-selective

            WARNING: Use this function if your resampled data is NOT normally
            distributed.

            Remember that this function can only tell you if a neuron is selective
            for a certain behavior or not. It will not classify what behavior the
            behavior the neuron was actually selective for.

            Args:
                dataframe: a Pandas DataFrame that contains the neuron(s) column
                vector(s) to be classified.
                resampled_df: the Pandas Dataframe of all the computed rates after
                resampling.

                real_diff_df: DataFrame

                    A Pandas DataFrame with one row that has the real D_hat,
                    difference of means, values.

               p_value: float, optional

                    The cutoff value for the probability that an effect could
                    occur by chance, default is 0.05.

               threshold: int, optional
                    The minimum required number of events for a neuron to be
                    considered classifiable, default is 10.

           Returns:
               classified_neurons: a dictionary of key-value pairs, where the name
               of each classified neuron is a key, and that neuron's classification
               is the corresponding value.
        """
        classified_neurons = dict()
        for neuron in dataframe.columns:
            if Resampler.get_num_of_events(dataframe, neuron) < threshold:
                if not neuron in classified_neurons:
                    classified_neurons[neuron] = "unclassified"

        for neuron in resampled_df.columns:
            if Resampler.compute_two_side_p_val(resampled_df, real_diff_df, neuron) <= p_value:
                if not neuron in classified_neurons:
                    classified_neurons[neuron] = "selective"
            else:
                if not neuron in classified_neurons:
                    classified_neurons[neuron] = "not-selective"

        return classified_neurons

    def compute_two_tailed_test(resampled_df, real_d_df, neuron, kwargs):
        """Classifies a given neuron via a two-tailed hypothesis test.

        WARNING: Use this function ONLY if your resampled data is indeed normally
        distributed.

        Classifies a given neuron as selective for a certain behavior, selective for
        when that behavior is not performed, or non-selective.
        Although one can use this as a stand alone function to classify a single
        neuron for a certain animal as either a <behavior> neuron, a
        "Not"-<behavior> neuron, or a "Non-selective" neuron, it is meant to be used
        a helper function for the classify_neurons_parametrically() function.

        Args:
            resampled_df: a resampled Pandas DataFrame with all the possible rates
            real_diff_df: a pandas DataFrame with one row that has the real
            difference of means values for a given animal and a corresponding
            behavior
            neuron: a single neuron of the neuron to classify use the two-tailed
            hypothesis test
            behavior_name: the behavior to classify the neuron by, e.g. "Running"
            or "Non-Running"
            high_tail: the cutoff for the upper-tail of the distribution
            low_tail: the cutoff for the lower-tail of the distribution

        Returns:
            behavior_name, Not-<behavior_name>, or "Not-selective"; based on the result of the
            two-tailed hypothesis test.
        """
        if real_d_df[neuron]['D'] >= np.percentile(resampled_df[neuron], kwargs["high"]):
            return kwargs["behavior1_name"]
        elif real_d_df[neuron]['D'] <= np.percentile(resampled_df[neuron], kwargs["low"]):
            return kwargs["behavior2_name"]

        return "not-selective"

    def normal_neuron_classifier(dataframe, resampled_df, real_diff_vals, **kwargs):
        """Classifies a given set of neurons

        This function simply calls is_neuron_selective for all the neurons
        for a given animal.

        Args:
            resampled_df: DataFrame

                The pandas DataFrame with all of the "simulated" D_hat values
                attained after resampling.

            real_diff_vals: DataFrame

                A Pandas DataFrame with one row that has the real difference of
                means.

            high: int, optional

                The cutoff for the upper-tail of the distribution, default is 95.

            low: int, optinonal

                The cutoff for the lower-tail of the distribution, default is 5.

            threshold: int, optional

                The minimum required number of events for a neuron to be considered
                classifiable, default is 10.

        Returns:
            classified_neurons: a dictionary of key-value pairs, where the name of
            each classified neuron is a key, and that neuron's classification is the
            corresponding value.
        """
        classified_neurons = dict()
        for neuron in dataframe.columns:
            threshold = kwargs.get("threshold", 10)
            if Resampler.get_num_of_events(dataframe, neuron) < threshold:
                if not neuron in classified_neurons:
                    classified_neurons[neuron] = "unclassified"

        for neuron in resampled_df.columns:
            if not neuron in classified_neurons:
                classified_neurons[neuron] = Resampler.compute_two_tailed_test(resampled_df, real_diff_vals, neuron, kwargs)

        return classified_neurons
