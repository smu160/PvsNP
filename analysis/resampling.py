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
    def get_num_of_events(dataframe, neuron):
        """Get the number of signal spikes for a given column vector

            Args:
                dataframe: DataFrame

                    A Pandas DataFrame that contains at least one neuron's 
                    signal data, in column vector form.
               
                neuron: 
                    The name of the neuron column vector to get the number
                    of events for.

           Returns:
               The amount of datapoints in a given column vector of nonzero
               value.
        """
        return len(dataframe.loc[:, neuron][dataframe[neuron] != 0])

    @staticmethod
    def compute_diff_rate(dataframe, neuron_col_names, *behaviors, frame_rate=10):
        """Computes difference between the rates of two behaviors

        Args:
            dataframe: DataFrame

                A concatenated pandas DataFrame of all the neuron column vectors
                for a given animal and its corresponding behavior column 
                vectors.

            neuron_col_names: list

                    The names of the neuron column vectors to be computed.

            *behaviors: str

                A single or ordered pair of behaviors to compute the difference 
                of means rate for, e.g. "OpenArms" vs. "ClosedArms".

            frame_rate: int

                The framerate associated with the given data; default is 10

        Returns: numpy array

            A numpy array of all the difference of means, D_hat, values, i.e. 
            all of the behavior vectors means subtracted from the corresponding 
            means of the non-behavior vectors, all scaled by the frame rate.
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
    def shuffle_worker(queue, resamples, neuron_concated_behavior, neuron_col_names, *behaviors):
        """Helper function for shuffle()
        
        This function repeats the permutation resampling and computation of 
        the test statistic (difference of means), a *resamples* amount of times.
        This allows us to create a permutation distribution for each neuron 
        column vector, under the condition that the null hypothesis is true, 
        i.e., that the neuron is not-selective for the behaviors.
        
        NOTE: This function is meant to be only be used as a helper function
        for the shuffle() function.

        Args:
            queue: Queue
            
                The blocking queue to which the resulting dataframe will be 
                added to.
            
            neuron_concated_behavior: DataFrame
            
                The neuron activity dataframe & the corresponding behavior 
                dataframe concatenated together, for a given animal.

            neuron_col_names: list
                
                The names of the neuron column vectors in the 
                neuron_concated_behavior dataframe.
                
            behaviors:
                
                The two behaviors, as strings, to be used as the two groups to 
                use for permutation resamples.
        """
        first_col = neuron_col_names[0]
        last_col = neuron_col_names[len(neuron_col_names)-1]
        
        rows_list = []
        for _ in range(resamples):
            neuron_concated_behavior.loc[:, first_col:last_col] = neuron_concated_behavior.loc[:, first_col:last_col].sample(frac=1).reset_index(drop=True)
            row = Resampler.compute_diff_rate(neuron_concated_behavior, neuron_col_names, behaviors[0], behaviors[1])
            rows_list.append(dict(zip(neuron_col_names, row)))

        queue.put(pd.DataFrame(rows_list, columns=neuron_col_names)) 
        
    @staticmethod
    def shuffle(resamples, neuron_concated_behavior, neuron_col_names, *behaviors):
        """Permutation resampling function for neuron selectivty analysis.

        This function simply starts a new process for each CPU that the machine
        has. More specifically, this function starts the shuffle_worker() 
        function for each new process, in order to allow the permutation 
        distribution for each neuron column vector to be created in a more 
        expedited fashion. More specifically, the amount of required permutation
        resamples is split evenly amongst all of the CPU's of the machine this
        function will be run on.

        Args:
            resamples: int 
            
                The total amount of permutation resamples desired.
            
            neuron_concated_behavior: DataFrame
            
                The neuron activity dataframe & the corresponding behavior 
                dataframe concatenated together, for a given animal.

            neuron_col_names: list
                
                The names of the neuron column vectors in the 
                neuron_concated_behavior dataframe.
                
            behaviors:
                
                The two behaviors, as strings, to be used as the two groups to 
                use for permutation resamples.
            
        Returns: 
        
            A (vertically) concatenated pandas DataFrame of all the DataFrames
            the shuffle_worker processes produced.
        """
        if len(behaviors) != 2:
            raise ValueError("You provided an appropriate amount of behaviors.")
        
        resamples_per_worker = resamples // os.cpu_count()
        queue = Queue()
        processes = []
        rets = []
        for _ in range(os.cpu_count()):
            process = Process(target=Resampler.shuffle_worker, args=(queue, resamples_per_worker, neuron_concated_behavior, neuron_col_names, *behaviors))
            processes.append(process)
            process.start()
        for process in processes:
            ret = queue.get()  # will block
            rets.append(ret)
        for process in processes:
            process.join()

        return pd.concat(rets, ignore_index=True)
   
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
    def non_normal_neuron_classifier(dataframe, resampled_df, real_diff_df, **kwargs):
        """Classify neurons as selective or not-selective

            WARNING: Use this function if your resampled data is NOT normally
            distributed.

            Remember that this function can only tell you if a neuron is 
            selective for a certain behavior or not. It will not classify what 
            behavior the behavior the neuron was actually selective for.

            Args:
                dataframe: DataFrame
                
                    A Pandas DataFrame that contains the neuron(s) column
                    vector(s) to be classified.
                
                resampled_df: DataFrame
                
                    A Pandas Dataframe of all the computed difference of mean 
                    values, D_hat, after permutation resampling. 

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
                classified_neurons: dictionary 
                
                A dictionary of key-value pairs, where the name of each 
                classified neuron is a key, and that neuron's classification 
                is the corresponding value.
        """
        p_value = kwargs.get("p_value", 0.05)
        threshold = kwargs.get("threshold", 10)

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

        WARNING: Use this function ONLY if your resampled data is indeed 
        normally distributed.

        Classifies a given neuron as selective for a certain behavior, selective
        for when that behavior is not performed, or non-selective.
        Although one can use this as a stand alone function to classify a single
        neuron for a certain animal as either a <behavior> neuron, a
        "Not"-<behavior> neuron, or a "Non-selective" neuron, it is meant to be
        used as a helper function for the normal_neuron_classifier() function.

        Args:
            resampled_df: DataFrame 
            
                A pandas DataFrame with all of the computed statistics for each
                permutation resample that was taken.
            
            real_d_df: DataFrame 
            
                A row of the actual D_hat, real difference of means, values for
                of each of the neuron column vectors.

            neuron: 
                
                The name of the neuron column vector to classify, using the 
                two-tailed hypothesis test.
            
            behavior1_name: str
            
                The behavior to classify the neuron by, e.g., "OpenArms".

            behavior2_name: str
            
                The behavior to classify the neuron by, e.g., "ClosedArms".

            high: float
            
                The cutoff for the upper-tail of the distribution.
            
            low: float
            
                The cutoff for the lower-tail of the distribution.

        Returns:
            behavior_name, Not-<behavior_name>, or "Not-selective"; based on the
            result of the two-tailed hypothesis test.
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

                The cutoff for the upper-tail of the distribution, 
                default is 95.

            low: int, optinonal

                The cutoff for the lower-tail of the distribution, default is 5.

            threshold: int, optional

                The minimum required number of events for a neuron to be 
                considered classifiable, default is 10.

        Returns:
            classified_neurons: dictionary

                A dictionary of key-value pairs, where the name of each 
                classified neuron is a key, and that neuron's classification 
                is the corresponding value.
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
