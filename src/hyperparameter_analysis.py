# hyperparameter_analysis.py

# Import public modules
import collections
import copy
import datetime
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class HyperparamAnalyzer(object):
    # Define the Unix time date string relative to which Unix time is measured.
    _UNIX_TIME_DATE_STR = '1970-01-01'

    def __init__(self, 
                 model_name, 
                 labels='any', 
                 output_base_dir='../outputs', 
                 metric_optimization='min',
                 optimal_metric_value='over_all_epochs',
                 date_range=[None, None], 
                 log10_hyperparams=[], 
                 equiv_hyperparams_dict={}, 
                 save_folder_name=None):
        """
        Analyze the influence of hyperparameters (hp) on the validation metric.

        Args:
            model_name (str, list of str, or None): Name of the trained model or a list of names of trained models.
                Remark: None should only be passed when initializing object from file.
            labels (list of str, 'any', or None): List of the labels that should be loaded for the model. In case that
                labels='any', all labels for the model will be loaded.
                Remark: None should only be passed when initializing object from file.
                (Default: 'any')
            output_base_dir (str): Path to the output directory in which the train_model.py outputs are saved in.
                (Default: '../outputs')
            metric_optimization (str): Flag that specifies if the metric should be either minimized ('min') or maximized ('max').
                (Default: 'min')
            optimal_metric_value (str): Flag that specifies if the optimal metric value should be either determined 
                over all epochs ('over_all_epochs') or taken as the metric value of the last epoch ('last_epoch').
                (Default: 'over_all_epochs')
            date_range (list or None): Two element list of the form [start_date, end_date] where the start/end date is either a 
                string with datetime_str_format='yyyy-mm-dd' or None (in which case the start date is 2022-01-01 and the 
                end date correspond to today).
                Remark: None should only be passed when initializing object from file.
                (Default: [None, None])
            log10_hyperparams (list): Specify for which hyperparamters the log10 value should be used in the analysis 
                (because their values differ in order of magnitudes)?
                (Default: [])
            equiv_hyperparams_dict (dict): Dictionary that has hyperparameters as dictionary-keys and a list of other hyperparameters 
                as dictionary-values in the form {key_hyperparam: equiv_hyperparams}. In case that the values of the columns associated
                with these hyperparameters ('key_hyperparams' and within 'equiv_hyperparams') are the same for all rows in 'self.raw_runs_df', 
                the columns of 'raw_runs_df' associated with the hyperparameters in 'equiv_hyperparams' will be considered 'redundant' and
                are therefore dropped from 'raw_runs_df' only keeping 'key_hyperparam'
                Use case: Imagine one hyperparameter is set to the value of another (e.g. global) hyperparameter in the config file, all
                the values of the two hyperparameters will be the same and it does not make sense to 'analyze' both.
                (Default: {})
            save_folder_name (str): Name of the folder in which the information/attributes of the initialized instance will be saved in.
                If None, this name will be generated.
                (Default: None)

        Remark: Within this class 'hyperparameter names' correspond to the flattened config dictionary keys.

        """
        # Parse model_name
        if model_name is None or isinstance(model_name, str):
            # In case that model_name is a string assign it to a class attribute of the same name
            self.model_name = model_name            
        else:
            # In case model_name is not a string, throw an error
            err_msg = f"The input 'model_name' must be either a string, got type '{type(model_name)}' instead."
            raise TypeError(err_msg)

        # Parse labels
        if labels is None:
            self.labels = labels
        elif isinstance(labels, str):
            # In case that the input labels is a string, differ the cases where it is any (in lower case) and assign 'any' to the class attribute self.labels
            if labels.lower()!='any':
                err_msg = f"If the input 'labels' is passed as string, it must correspond to 'any' (in lower case), got value '{labels}' instead."
                raise ValueError(err_msg)
            self.labels = 'any'
        elif isinstance(labels, list):
            # In case that the input labels is a list assign, check that all its entries are strings and if so, assign the input to the class attribute self.labels
            for label in labels:
                if not isinstance(label, str):
                    err_msg = f"The elements of the list input 'labels' must all be integers, got element '{label}' of type '{type(label)}' instead."
                    raise TypeError(err_msg)

            self.labels = labels
        else:
            # In case labels is neither a string nor a list (of strings), throw an error
            err_msg = f"The input 'labels' must be either a string (equal to 'any') or a list of strings (list containing the to be included labels), got type '{type(labels)}' instead."
            raise TypeError(err_msg)

        # Assign inputs to class attributes
        self.output_base_dir        = output_base_dir
        self.metric_optimization    = metric_optimization
        self.optimal_metric_value   = optimal_metric_value
        self.log10_hyperparams      = log10_hyperparams
        self.equiv_hyperparams_dict = equiv_hyperparams_dict

        # Check that the output base directory exists
        if not os.path.isdir(self.output_base_dir):
            err_msg = f"The directory '{self.output_base_dir}' does not exist."
            raise FileNotFoundError(err_msg)

        # Ensure that metric optimization is one of the expected values
        if self.metric_optimization not in ['min', 'max']:
            err_msg = f"The 'metric_optimization' must be either 'min' (minimize the metric) or 'max' (maximize), got '{self.metric_optimization}' instead."
            raise ValueError(err_msg)

        # Ensure that optimal_metric_value is one of the expected values
        if self.optimal_metric_value not in ['over_all_epochs', 'last_epoch']:
            err_msg = f"The 'optimal_metric_value' must be either 'over_all_epochs' (optimal metric value determined over all epochs) or 'last_epoch' (use the metric value of the last epoch as the optimal metric value), got '{self.optimal_metric_value}' instead."
            raise ValueError(err_msg)

        # Parse the input date range and assing it to a class attribute of the same name
        if date_range is None:
            self.date_range = None
        else:
            self.date_range = self.parse_date_range(date_range)

        # In case that the 'save_folder_name' is not passed (None) construct it from the date range and the model names
        if save_folder_name is None:
            # Construct the labels info to be used for the save folder name, while differing
            # the cases where self.labels is a list or not
            if isinstance(self.labels, list):
                # Differ the cases where the list has only one element or more than one element
                if len(self.labels)==1:
                    # The labels information correspond to the single element
                    labels_info = self.labels[0]
                else:
                    # In case there are multiple labels, join them with ',' and put the result in brackets
                    labels_info = f"[{','.join(self.labels)}]"
            else:
                labels_info = self.labels

            save_folder_name = f"{self.model_name}:{labels_info}:[{self.date_range[0]},{self.date_range[1]}]"
        self.save_folder_name = save_folder_name

        # Define a map from 'easy to work with' hyperparameter names to their corresponding 'hyperparameter name' 
        # that corresponds to a flattened configuration dictionary key
        self.hyperparam_name_map = {
            'num_cv_folds':  'data_handling.data_preprocessing.K', # Number of cross validation folds
            'cv_fold_index': 'data_handling.data_preprocessing.k', # Index of cross validation folds
            'num_epochs':    'training.num_epochs', # Number of epochs
        }

        # Specify which columns contain iterables in each of their cells
        # E.g. the 'epochs' column contains a tuple in the cell of each row (corresponding to one hyperparameter combination)
        self.iterable_col_names = list()
        self.iterable_col_names += ['train_epochs', 'train_metric_values', 'mean_train_metric_values', 'std_train_metric_values', 'min_train_metric_values', 'max_train_metric_values']
        self.iterable_col_names += ['valid_epochs', 'valid_metric_values', 'mean_valid_metric_values', 'std_valid_metric_values', 'min_valid_metric_values', 'max_valid_metric_values']
        self.iterable_col_names += ['learning_rate_epochs', 'learning_rate_values']

        # Define which attributes should be saved
        self.save_attributes = ['model_name', 'labels', 'date_range', 'hyperparam_names_set', 'metric_optimization', 'equiv_hyperparams_dict', 'problematic_runs', 'missing_args_dict_list']

        # Define the relative path to the job management directory
        self.job_management_dir = '../job_management'

        # Define the base directory where folders will be saved in that each hold the information/attributes of different class attributes
        self.save_base_dir = str( Path('./saved/', 'hyperparam_analysis_files') )

        # Construct the save folder path
        self.save_folder_path = str( Path(self.save_base_dir, self.save_folder_name) )

        # Define the base directory where argument combination files will be saved in
        self.args_combinations_files_dir_path = str( Path(self.job_management_dir, 'args_combinations') )
        
        # Define a map from quantity labels to the corresponding file names
        self.quantity_label_to_file_names_map = {
            'save_attr_dict':   'saved_attributes.pickle',
            'combined_runs_df': 'combined_runs_df.tsv',
            'raw_runs_df':      'raw_runs_df.tsv',
        }

        # Initialize certain attributes to None
        self.problematic_runs       = None
        self.hyperparam_names_set   = None
        self.raw_runs_df            = None
        self.missing_args_dict_list = None
        self.combined_runs_df       = None

        print(f"Initialization completed, either construct the (raw and combined) DataFrames from the training output files '.construct()' or load them '.load()' (if they have already beend constructed and saved).\n")

    @classmethod
    def from_file(cls, 
                  file_name,
                  metric_optimization='min', 
                  optimal_metric_value='over_all_epochs',
                  log10_hyperparams=[], 
                  equiv_hyperparams_dict={}):
        """
        Load the hyperparameter analyzer from a file that contains the information
        of a previously constructed and saved hyperparameter analyzer..

        Args:
            file_name (str): Name of the file (in the folder './saved/hyperparam_analysis_files') 
                from which they hyperparameter analyzer should be loaded.
            metric_optimization (str): Flag that specifies if the metric should be either minimized ('min') or maximized ('max').
                (Default: 'min')
            optimal_metric_value (str): Flag that specifies if the optimal metric value should be either determined 
                over all epochs ('over_all_epochs') or taken as the metric value of the last epoch ('last_epoch').
                (Default: 'over_all_epochs')
            log10_hyperparams (list): Specify for which hyperparamters the log10 value should be used in the analysis 
                (because their values differ in order of magnitudes)?
                (Default: [])
            equiv_hyperparams_dict (dict): Dictionary that has hyperparameters as dictionary-keys and a list of other hyperparameters 
                as dictionary-values in the form {key_hyperparam: equiv_hyperparams}. In case that the values of the columns associated
                with these hyperparameters ('key_hyperparams' and within 'equiv_hyperparams') are the same for all rows in 'self.raw_runs_df', 
                the columns of 'raw_runs_df' associated with the hyperparameters in 'equiv_hyperparams' will be considered 'redundant' and
                are therefore dropped from 'raw_runs_df' only keeping 'key_hyperparam'
                Use case: Imagine one hyperparameter is set to the value of another (e.g. global) hyperparameter in the config file, all
                the values of the two hyperparameters will be the same and it does not make sense to 'analyze' both.
                (Default: {})

        Return:
            (HyperparamAnalyzer): A hyperparameter analyzer object loaded from the file.
        
        """
        # Initialize the object
        hyperparam_analyzer = HyperparamAnalyzer(model_name=None, # Parsed from to be loaded file 
                                                 labels=None, # Parsed from to be loaded file
                                                 metric_optimization=metric_optimization,
                                                 optimal_metric_value=optimal_metric_value,
                                                 date_range=None, # Parsed from to be loaded file 
                                                 log10_hyperparams=log10_hyperparams, 
                                                 equiv_hyperparams_dict=equiv_hyperparams_dict, 
                                                 save_folder_name=file_name)

        # Load the objects content from file and return it
        hyperparam_analyzer.load()
        return hyperparam_analyzer
    
    def construct(self):
        """ Construct the (raw and combined) DataFrames from the training output files. """
        # Construct the pandas.DataFrames holding the information about the different runs
        self.raw_runs_df = self.construct_raw_runs_df()

        # Generate the missing arguments dictionary list
        self.missing_args_dict_list = self.generate_missing_args_dict_list()

        # Combine the information for all runs that were performed for the same hyperparameter values combination 
        # but for different cross validation folds
        # Remark: Actually the cross validation fold index are hyperparameters.
        #         So to be very precise: the combination is done for all hyperparameter values combination not
        #         including the cross validation fold index as hyperparameter
        self.combined_runs_df = self.generate_combined_runs_df()

        # In case that self.combined_runs_df is empty, throw an error
        if len(self.combined_runs_df)==0:
            err_msg = f"The combined_runs DataFrame is empty."
            raise ValueError(err_msg)

        # Save the information of the class instance
        self.save()

    def save(self):
        """ Save specific 'attributes' of the instance. """
        print("\nSaving information/attributes of the current class instance.")

        # Generate the folder (in which the different attributes will be saved in as files) if it doesn't 
        # already exist
        if not os.path.isdir(self.save_folder_path):
            os.makedirs(self.save_folder_path)
            print(f"Created the folder {self.save_folder_path}")

        ################################################################################################################
        ### 1) SAVE SPECIFIC ATTRIBUTES AND CHECK THAT LOADING THE SAVED FILE LEADS TO THE SAME OBJECT
        #######################################################################################################################
        # 1a)
        # Generate a dictionary from the values of certain class attributes and save it as pickle file
        save_attr_dict  = dict()
        for save_attribute in self.save_attributes:
            # Get the value of the current attribute and assign it as dictionary-value
            # to the save attribute as key
            save_attr_dict[save_attribute] = getattr(self, save_attribute)

        # Save the quantity
        self.save_quantity('save_attr_dict', save_attr_dict)
        
        # 1b) Load the saved attributes and check its equivalence to the original dictionary
        loaded_dict = self.load_quantity('save_attr_dict')
        if loaded_dict!=save_attr_dict:
            err_msg = f"Loaded the attributes leads to a dictionary that is not equivalent to the original dictionary!"
            raise ValueError(err_msg)
        #######################################################################################################################

        #######################################################################################################################
        ### 2) SAVE THE COMBINED RUNS DATAFRAME AND CHECK THAT LOADING THE SAVED FILE LEADS TO THE SAME OBJECT
        #######################################################################################################################
        # 2a) Save the 'combined runs' DataFrame
        self.save_quantity('combined_runs_df', self.combined_runs_df)

        # 2b) Load the 'combined runs' DataFrame and check its equivalence to the original DataFrame
        loaded_df = self.load_quantity('combined_runs_df')
        self.check_dataframe_equivalence(loaded_df, self.combined_runs_df, rel_eps=1e-4)
        #######################################################################################################################

        #######################################################################################################################
        ### 3) SAVE THE RAW RUNS DATAFRAME AND CHECK THAT LOADING THE SAVED FILE LEADS TO THE SAME OBJECT
        #######################################################################################################################
        # 3a) Saving
        # Save the 'raw runs' DataFrame
        self.save_quantity('raw_runs_df', self.raw_runs_df)

        # 3b) Load the 'raw runs' DataFrame and check its equivalence to the original DataFrame
        loaded_df = self.load_quantity('raw_runs_df')
        self.check_dataframe_equivalence(loaded_df, self.raw_runs_df, rel_eps=1e-4)
        #######################################################################################################################

        print('Saving done.\n')

    def load(self):
        """ Load a previously saved instance of the class saved in self.save_folder_name. """
        # Check that this folder exists and throw an error otherwise
        if not os.path.isdir(self.save_folder_path):
            err_msg = f"Can not load from the folder '{self.save_folder_path}' because it doesn't exist."
            raise FileNotFoundError(err_msg)

        print(f"\nLoad instance information from the folder {self.save_folder_path}")

        # Load the saved attributes (not all but only specific attributes) as dictionary
        loaded_attr_dict = self.load_quantity('save_attr_dict')

        # Loop over its key-value pairs
        for attr_name, loaded_attr_value in loaded_attr_dict.items():
            # Get the attribute value of the current instance
            attr_value = getattr(self, attr_name)

            # Differ the cases where the attribute value is None or not
            if attr_value is None:
                # If the attribute value is None, assign the loaded attribute value to the attribute
                setattr(self, attr_name, loaded_attr_value)
                print(f"Loaded the values of the attribute '{attr_name}'")
            else:
                # Otherwise, if the attribut value is not None check that it is equivalent to the loaded value
                # and throw an error otherwise
                if attr_value!=loaded_attr_value:
                    err_msg = f"The attribute value and the loaded attribute value are not the same for the attribute '{attr_name}'.\nThe attribute value is: {attr_value}\nThe loaded attribute value is: {loaded_attr_value}"
                    raise ValueError(err_msg)

        # Load the raw runs and combined runs DataFrames and assign them to the corresponding class attributes
        self.combined_runs_df = self.load_quantity('combined_runs_df')
        self.raw_runs_df      = self.load_quantity('raw_runs_df')

        print(f"Loading done\n")

    @property
    def non_redundant_hyperparam_names(self):
        """ Return a list of all non-redundant hyperparameter names. """
        # Determine the column names of the raw_runs DataFrame
        col_names = self.raw_runs_df.columns

        # The non-redundant hyperparameters are the column names that are also 
        # hyperparameter names (that correspond to flattened config dictionary keys).
        # Thus return intersection between the set of column names (of 'self.raw_runs_df') 
        # and the set of hyperparameter names, casted to a list
        return list( self.hyperparam_names_set.intersection(set(col_names)) )

    @property
    def cross_validation_hyperparam_names(self):
        """ Return a list of the hyperparameter names associated with cross validation. """
        return [self.hyperparam_name_map['num_cv_folds'], self.hyperparam_name_map['cv_fold_index']]

    def parse_date_range(self, 
                         date_range):
        """
        Parse the date range. 
        
        Args:
            date_range (list): Two element list of the form [start_date, end_date] where the start/end date is either a 
                string with datetime_str_format='yyyy-mm-dd' or None (in which case the start date is 2022-01-01 and the 
                end date correspond to today).

        Return:
            ([datetime.datetime, datetime.datetime]): List containing two datetime objects with the form:
                '[start_date, end_date]'.

        """
        # Parse the start date (first element of date_range)
        if date_range[0] is None:
            # If the start date is None, use 'self._UNIX_TIME_DATE_STR' as start date
            start_date = self.cast_str_to_date_object(self._UNIX_TIME_DATE_STR)
        else:
            # If the start date is not None, assume it is a string in the correct form 
            # and cast it to a date object
            start_date = self.cast_str_to_date_object(date_range[0])

        # Parse the end date (second element of date_range)
        if date_range[1] is None:
            # If the end date is None, use 'today' to generate a date object
            end_date = datetime.date.today()
        else:
            # If the end date is not None, assume it is a string in the correct form 
            # and cast it to a date object
            end_date = self.cast_str_to_date_object(date_range[1])

        # Return a list containing the datetime objects corresponding to the start and end date
        return [start_date, end_date]

    def cast_str_to_date_object(self, 
                                date_str):
        """
        Return the date object corresponding to the input string.

        Args:
            date_str (str): String of the date in the form 'yyyy-mm-dd'.

        Return:
            (date object): Date object corresponding to the string.
        
        """
        # Try to create a datetime object from the str
        try:
            datetime_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            # In case that a ValueError is thrown, catch it and throw a custom error
            err_msg = f"The input '{date_str}' is not a string of the form 'yyyy-mm-dd' and can therefore not be cast to a datetime object."
            raise StringNotCastableToDateError(err_msg)

        # Generate the date object from the datetime object and return it
        return datetime_obj.date()
        
    def construct_raw_runs_df(self):
        """ 
        Constuct a DataFrame whose rows hold the information for the various hyperparameters combinations as well as the train metric
        values and associated epochs.
        """
        # Inform the user which files are loaded
        self.display_loading_info()

        # Initialize a list that will be filled with all the run folder paths of problematic runs
        self.problematic_runs = list()

        # Initialize a set that will be filled with all the hyperparameter names
        self.hyperparam_names_set = set()
        
        # Loop over the 'date folders' in the output base directory
        # Remark: The folder structure is '<output_base_dir>/<date>/<model_name>/<run_folder>'.
        #         The <run_folder> is a folder that corresponds to a specific set of overriden hyperparameters
        #         and contains the associated config file and the parameter checkpoints etc.
        raw_runs_dict = collections.defaultdict(list)
        for date_folder in os.listdir(self.output_base_dir):
            # Try to cast the name of the date folder to a date object and if a 'StringNotCastableToDateError' the folder 
            # name does not have the correct date format (i.e. it is not an actual date folder as assumed up to here) and 
            # thus should be skipped. Continue to the next date folder in this case.
            try:
                date_obj = self.cast_str_to_date_object(date_folder)
            except StringNotCastableToDateError:
                continue

            # In case that the date object is not within the date range continue to the next folder
            if not (self.date_range[0]<=date_obj and date_obj<=self.date_range[1]):
                continue
            
            # Construct the path to the date folder
            date_folder_path = str( Path(self.output_base_dir, date_folder) )

            # Loop over the 'model folders' in the current 'date folder'
            for model_folder in os.listdir(date_folder_path):
                # In case that the model_folder name does not correspond to the model name, continue to the next model folder
                if model_folder!=self.model_name:
                    continue
            
                # Construct the base directory to the model's training outputs resulting for configurations 
                # obtained for the date (of the current iteration) and the specified model
                # Remark: The model_folder_path corresponds to '<output_base_dir>/<date>/<model_name>'
                model_folder_path = str( Path(date_folder_path, model_folder) )

                # Loop over the 'label folders' in the current 'model folder'
                for label_folder in os.listdir(model_folder_path):
                    # In case that only specific labels should be considered (self.labels is not 'any'), check
                    # that the current label corresponds to one of the required labels and continue to the next 
                    # label folder in case it doesn't
                    if self.labels!='any':
                        if label_folder not in self.labels:
                            continue
                
                    # Construct the base directory to the model's training outputs resulting for configurations 
                    # obtained for the date (of the current iteration) and the specified model
                    # Remark: The label_folder_path corresponds to '<output_base_dir>/<date>/<model_name>/<label>'
                    label_folder_path = str( Path(model_folder_path, label_folder) )

                    # Loop over all the 'run folders' (output for a specific run) in the current 'label folder'
                    for run_folder in os.listdir(label_folder_path):
                        # Skip the '.ipynb_checkpoints' folder
                        if run_folder=='.ipynb_checkpoints':
                            print(f"Skipped '.ipynb_checkpoints' folder:\n{run_folder_path}\n")
                            continue
                            
                        # Generate the path to the current run folder
                        # Remark: The <run_folder> is a folder that corresponds to a specific set of overriden hyperparameters
                        #         and contains the associated config file and the parameter checkpoints etc.
                        run_folder_path = str( Path(label_folder_path, run_folder) )

                        # Get the flattened config dictionary for the current run
                        # Remark: The method 'get_flattened_config_dict' returns None in case no config file exists in the run folder.
                        #         If this is the case, add the run folder path to the problematic runs, inform the user, 
                        #         and continue to the next run
                        flattened_config_dict = self.get_flattened_config_dict(run_folder_path)
                        if flattened_config_dict is None:
                            self.problematic_runs.append(run_folder_path)
                            print(f"No config file found in the run folder:\n{run_folder_path}\n")
                            continue

                        # Get a dictionary containing the tracked quantities (e.g. training loss, train metric, validation metric)
                        # over several epochs for the current run.
                        # Remark: The method 'get_tracked_quantities_dict' returns None in case no tracked quantities file exists in the run folder.
                        #         If this is the case, add the run folder path to the problematic runs, inform the user, 
                        #         and continue to the next run
                        tracked_quantities_dict = self.get_tracked_quantities_dict(run_folder_path)
                        if tracked_quantities_dict is None:
                            self.problematic_runs.append(run_folder_path)
                            print(f"No checkpoints (i.e. tracked quantities) found in the run folder:\n{run_folder_path}\n")
                            continue

                        # Update the set of hyperparameter names
                        # Remark: The flattened config dictionary keys are considered to be the 'hyperparameter names'
                        self.hyperparam_names_set = self.hyperparam_names_set.union( set(flattened_config_dict.keys()) )

                        # Extract the train metric values and corresponding epochs from the tracked quantities dictionary
                        train_epochs, train_metric_values = self.get_epochs_and_metric_values(tracked_quantities_dict, set_name='train')
                        
                        # Extract the validation metric values and corresponding epochs from the tracked quantities dictionary
                        valid_epochs, valid_metric_values = self.get_epochs_and_metric_values(tracked_quantities_dict, set_name='valid')

                        # Extract the learning rate values and corresponding epochs from the tracked quantities dictionary
                        learning_rate_epochs, learning_rate_values = self.get_epochs_and_tracked_quantity_values(tracked_quantities_dict, quantity_name='learning_rate')

                        # Get the number of epochs the model was supposed to be trained for
                        num_epochs = flattened_config_dict[self.hyperparam_name_map['num_epochs']]

                        # If the last entry in 'train_epochs' corresponds to this number, add the run folder path to the problematic runs,
                        # inform the user, and continue to the next folder
                        if num_epochs!=train_epochs[-1]:
                            self.problematic_runs.append(run_folder_path)
                            print(f"The number of epochs the model should have been trained for is '{num_epochs}' but the model has only been trained for '{train_epochs[-1]}' epochs for the run folder:\n{run_folder_path}\n => Skip this run in the construction of 'raw_runs_df'!\n")
                            continue

                        # Append the hyperparams values to the corresponding lists in the raw_runs dictionary
                        # Remark: Remember, the flattened config dictionary keys are considered as 'hyperparameter names'
                        for key, value in flattened_config_dict.items():
                            raw_runs_dict[key].append(value)

                        # Append the path to the current run folder
                        raw_runs_dict['run_folder_path'].append(run_folder_path)

                        # Append the validation metric values and corresponding epochs
                        raw_runs_dict['train_epochs'].append(train_epochs)
                        raw_runs_dict['train_metric_values'].append(train_metric_values)

                        # Append the validation metric values and corresponding epochs
                        raw_runs_dict['valid_epochs'].append(valid_epochs)
                        raw_runs_dict['valid_metric_values'].append(valid_metric_values)

                        # Append the learning rate values and corresponding epochs
                        raw_runs_dict['learning_rate_epochs'].append(learning_rate_epochs)
                        raw_runs_dict['learning_rate_values'].append(learning_rate_values)

        # Check that at least one file was found for one combination hyperparameter value combination (and thus raw_runs_dict
        # contains at least one element). Throw an error if this is not the case.
        if len(raw_runs_dict)==0:
            err_msg = f"No output files were found in the time from {self.date_range[0]} to {self.date_range[1]} for the model '{self.model_name}' with '{self.labels}' labels in the output directory '{self.output_base_dir}'."
            raise ValueError(err_msg)

        # Map the raw_runs dictionary to a dataframe
        raw_runs_df = pd.DataFrame(raw_runs_dict)

        # Loop over all columns of the raw_runs_df and if they contain list entries (that are unhashable) cast these entries to tuples
        for col_name in raw_runs_df.columns:
            raw_runs_df[col_name] = raw_runs_df[col_name].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # Loop over all columns of the raw_runs_df and if a column contains at least one string entry transform 
        # all other entries in the column to strings
        for col_name in raw_runs_df.columns:
            # Determine if the current column contains any string entry
            col_contains_strings = np.any([ isinstance(item, str) for item in raw_runs_df[col_name] ])
            
            # If the column contains any string entry, transform all entries of this column to strings
            if col_contains_strings:
                raw_runs_df[col_name] = raw_runs_df[col_name].apply(lambda x: str(x))

        # Remove the redundant columns in the DataFrame, drop duplicates and return it
        print(f"Loading done.")
        raw_runs_df = self.remove_redundant_columns(raw_runs_df)
        return raw_runs_df.drop_duplicates()

    def display_loading_info(self):
        """ Display the loading information. """
        # Construct the label information
        # Remark: Treat the case where the label is 'any' or a list of labels
        if self.labels=='any':
            label_info = f"'any' label"
        else:
            # In this case labels is a list of (string) labels
            label_info = f"the labels {self.labels}"

        # Construct the date_range information
        if self.date_range[0]==self.date_range[1]:
            # Start and end date are the same
            date_range_info = f"on {str(self.date_range[0])}"
        elif self.date_range[0]==self.cast_str_to_date_object(self._UNIX_TIME_DATE_STR):
            # In case no start date has been specified, Unix time is used as start date,
            # so there is actually no real start date
            date_range_info = f"until (and including) {str(self.date_range[1])}"
        else:
            date_range_info = f"from {str(self.date_range[0])} to {str(self.date_range[1])}"

        # Display the information
        print(f"Load training output files for the model {self.model_name} with {label_info} generated {date_range_info}, please wait...\n")

    def get_flattened_config_dict(self, 
                                  run_folder_path):
        """
        Return the flattened config dictionary for a specific run.

        Args:
            run_folder_path (str): Path to the run folder.

        Return:
            (dict or None): Flattened config dictionary or None in case that the config file does not exist.

        """
        # Generate the path to the config file 'config.yaml' located in the run folder
        config_file_path = str( Path(run_folder_path, 'config.yaml') )

        # Return None in case that the config file does not exist
        if not os.path.isfile(config_file_path):
            return None

        # Load the config file as config dictionary
        with open(config_file_path) as file:
            # Load the config file content as a dictionary
            config_dict = yaml.load(file, Loader=yaml.FullLoader) # This syntax works with 'pyyaml-6.0', but earlier pyyaml version did not use the 'Loader' keyword.

        # Flatten the config dict which is a nested dictionary and return it
        return self.flatten_nested_dict(config_dict)

    def flatten_nested_dict(self, 
                            nested_dict):
        """
        Flatten a nested dictionary.

        Args:
            nested_dict (dict): A potentially nested dictionary (=dictionary that contains itself dictionaries as values).

        Returns:
            (dict): A flattened dictionary (=dictionary that does not contain any dictionaries as values). The keys of any
                dictionary values in the original dictionary ('nested_dict') become new keys of the form '<key>.<sub_key>'.
        
        Example: The nested dictionary {'a': {'b': c}} is flattened to {'a.b': c}.

        """
        flattened_dict = dict() 
        for key, value in nested_dict.items():
            # Differ cases where the value is a dictionary itself or not
            if isinstance(value, dict):
                # In case that the value is itself a dictionary it might
                # be also itself a nested dictionary, so recursively call
                # the function itself on the value
                sub_dict = self.flatten_nested_dict(value)

                # Loop over the sub dictionary and add its key-value pairs 
                # to the flattened dictionary constructing the new key as a 
                # combination of the original key ('key') and the key ('sub_key) 
                # of each iterated key-value pair.
                for sub_key, sub_value in sub_dict.items():
                    flattened_dict[f"{key}.{sub_key}"] = sub_value
            else:
                # In case that the value is not itself a dictionary, add
                # it as value with the same key to the flattened dictionary
                flattened_dict[key] = value

        return flattened_dict

    def get_tracked_quantities_dict(self, 
                                    run_folder_path):
        """
        Return the tracked quantities dictionary -- containing the tracked quantities (e.g. training loss, training metric, validation metric)
        as a function of various epochs -- for a specific run.

        Args:
            run_folder_path (str): Path to the run folder.

        Return:
            (dict or None): Dictionary containing the tracked quantities or None in case that the checkpoints directory does not exist
                or is empty.
            
        """
        # Generate the path to the checkpoints directory of the current hyperparameters
        checkpoints_dir_path = str( Path(run_folder_path, 'checkpoints') )

        # Return None in case that the checkpoints directory does not exist
        if not os.path.isdir(checkpoints_dir_path):
            return None

        # Return None in case that the checkpoints folder is empty
        if len(os.listdir(checkpoints_dir_path))==0:
            return None
        
        # Construct a list of the epochs for which checkpoints were taken
        # Remark: These epoch-checkpoints folders have the form 'epoch_{epoch}'
        #         so that the epochs can be obtained by splitting the folder name
        #         at '_' and casting the resulting seconds element to an integer.
        checkpoint_epochs = [int(item.split('_')[1]) for item in os.listdir(checkpoints_dir_path)]

        # Construct the name of the checkpoints epoch folder taken for the maximal epoch
        max_checkpoint_epochs          = max(checkpoint_epochs)
        max_checkpoint_epochs_dir_name = f"epoch_{max_checkpoint_epochs}"

        # Construct the file path to the tracked quantities file
        tracked_quantities_file_path = str( Path(checkpoints_dir_path, max_checkpoint_epochs_dir_name, 'tracked_quantities.pickle') )

        # Throw an error if the file does not exist
        if not os.path.isfile(tracked_quantities_file_path):
            err_msg = f"The file '{max_checkpoint_epochs_dir_name}' does not exist."
            raise FileNotFoundError(err_msg)

        # Load the tracked quantities dictionary from the pickle data
        with open(tracked_quantities_file_path, 'rb') as file:
            try:
                tracked_quantities_dict = pickle.load(file)
            except EOFError:
                return None
                
        return tracked_quantities_dict

    def get_epochs_and_metric_values(self, 
                                     tracked_quantities_dict, 
                                     set_name='valid'):
        """
        Return the metric values and their corresponding epochs extracted from the tracked quantities dictionary.

        Args:
            tracked_quantities_dict (dict): Dictionary holding the tracked quantities (e.g. training loss, training metric,
                and validation metric) as a function of various epochs.
                It has the form {key: [(epoch_1, quantity_value_1), (epoch_, quantity_value_2)]} where the key corresponds to the 
                tracked quantities.
            set_name (str): Name of the set on which the metric was evaluated on and thus should be either 'train' or 'valid',
                (Default: 'valid')

        Returns:
            (tuple, tuple): Epochs and metric values as tuples.
        """
        # Get the epochs and tracked quantity values for the metric of the specified set
        return self.get_epochs_and_tracked_quantity_values(tracked_quantities_dict, quantity_name=f"metric_{set_name}")

    def get_epochs_and_tracked_quantity_values(self, 
                                               tracked_quantities_dict, 
                                               quantity_name):
        """
        Return the metric values and their corresponding epochs extracted from the tracked quantities dictionary.

        Args:
            tracked_quantities_dict (dict): Dictionary holding the tracked quantities (e.g. training loss, training metric,
                and validation metric) as a function of various epochs.
                It has the form {key: [(epoch_1, quantity_value_1), (epoch_, quantity_value_2)]} where the key corresponds to the 
                tracked quantities.
            quantity_name (str): Name of the tracked quantity for which their values (and corresponding epochs) should be returned.

        Returns:
            (tuple, tuple): Epochs and quantity values as tuples.
        """
        # Check that the quantity was tracked
        if not quantity_name in tracked_quantities_dict:
            err_msg = f"The quantity '{quantity_name}' was not stored in the 'tracked_quantities_dictionary'."
            raise ValueError(err_msg)

        # Get the a list of (epoch, quantity_values)-pairs from the dictionary-value to the dictionary-key quantity_name
        pair_list = tracked_quantities_dict[quantity_name]

        # Transform the pair list into two tuples, one for the epochs and one for the metric values
        epochs          = tuple([item[0] for item in pair_list])
        quantity_values = tuple([item[1] for item in pair_list])

        return epochs, quantity_values

    def remove_redundant_columns(self, 
                                 raw_runs_df):
        """
        Remove all redundant columns (columns that contain the same value in each row) in the passed DataFrame.

        Args:
            raw_runs_df (pandas.DataFrame): DataFrame, which contains the information of all the different runs 
                as rows, and whose redundant columns (that correspond for example to hyperparameter names) should 
                be removed.

        Return:
            (pandas.DataFrame): Input DataFrame with the redundant columns removed.
        """
        # Initialize the redundant columns to None
        redundant_cols = None

        ##############################################################################################################################
        # Step 1:
        # Find columns that contain only one entry in each cross validation fold
        # Remark: The value can be different for different cross validation folds but is the 
        #         same for each row for one cross validation fold
        ##############################################################################################################################
        # Get all the cross validation folds and loop over them
        # Remark: 'self.hyperparam_name_map' maps certain 'easy to work with' hyperparameter names to the corresponding 
        #         'hyperparameter names' that correspond to flattened config dictionary keys
        cv_fold_indices = list(set( raw_runs_df[self.hyperparam_name_map['cv_fold_index']] ))
        for cv_fold_index in cv_fold_indices:
            # Get the combinations DataFrame for the current cross validation fold
            cv_fold_raw_runs_df = raw_runs_df[raw_runs_df[self.hyperparam_name_map['cv_fold_index']]==cv_fold_index]

            # Loop over the columns of raw_runs_df and collect all columns that contain exactly
            # the same entry for all of their rows (=col_values). These columns are redundant.
            cv_fold_redundant_cols = list()
            for col_name in raw_runs_df.columns:
                # In case that the current column is not a hyperparameter name, continue to the next column
                if col_name not in self.hyperparam_names_set:
                    continue

                # In case that the current column corresponds to the hyperparameter name of the cross validation fold 
                # index or of the number of cross validation folds, continue to the next column
                if col_name in [self.hyperparam_name_map['cv_fold_index'], self.hyperparam_name_map['num_cv_folds']]:
                    continue

                # Get all the entries/values of the current column
                col_values = cv_fold_raw_runs_df[col_name]

                # Get the number of unique column values
                try:
                    num_unique_col_values = len(np.unique(col_values))
                except TypeError:
                    # In case there is a type error (e.g. in case that there are numeric and string entries) transform
                    # the column values to strings before determining the unique
                    num_unique_col_values = len(np.unique([str(item) for item in col_values]))

                # If this number is 1, add the column name to redundant columns of the current fold
                if num_unique_col_values==1:
                    cv_fold_redundant_cols.append(col_name)

            # Cast cv_fold_redundant_cols to a set
            cv_fold_redundant_cols = set(cv_fold_redundant_cols)

            # In case that redundant_cols is None (so it has not been defined before), assign cv_fold_redundant_cols to it
            if redundant_cols is None:
                redundant_cols = cv_fold_redundant_cols
            else:
                # Otherwise, update redundant_cols by taking the intersection of it with the current fold's set of redundant columns
                redundant_cols = redundant_cols.intersection(cv_fold_redundant_cols)

        ##############################################################################################################################

        ##############################################################################################################################
        # Step 2:
        # In case there are equivalent hyperparameters, check that the entries in their corresponding
        # columns are all the same and if so, remove the 'redundant' hyperparameters.
        # For this use self.equiv_hyperparams_dict that has the form '{key_hyperparam: [equiv_hyperparams]}'
        # In case that the column entries of 'key_hyperparam' and all the column entries of the hyperparameters 
        # in the list 'equiv_hyperparams' are the same, remove/drop the columns associated with the hyperparameters 
        # in hyperparams_list only keeping the hyperparameter 'key_hyperparam'.
        ##############################################################################################################################
        # Loop over the key-value pairs of self.equiv_hyperparams_dict
        redundant_hyperparam_names = list()
        for key_hyperparam_name, equiv_hyperparam_names in self.equiv_hyperparams_dict.items():
            # Check that key_hyperparam_name is a hyperparameter and throw an error otherwise
            if key_hyperparam_name not in self.hyperparam_names_set:
                err_msg = f"The key '{key_hyperparam_name}' of the passed dictionary 'equiv_hyperparams' is not the name of a hyperparameter. The valid hyperparameter names are:\n{list(self.hyperparam_names_set)}\n"
                raise ValueError(err_msg)

            # Get the values associated with the hyperparam 'key_hyperparam' as numpy arrays
            key_hyperparam_values = raw_runs_df[key_hyperparam_name].to_numpy()

            # Loop over the hyperparameters in the hyperparams list
            for equiv_hyperparam_name in equiv_hyperparam_names:
                # Get the values associated with the hyperparam 'equiv_hyperparam' as numpy arrays
                equiv_hyperparam_values = raw_runs_df[equiv_hyperparam_name].to_numpy()

                # Check if all entries of the numpy arrays 'key_hyperparam_values' and 'equiv_hyperparam_values' are the same
                if np.all(key_hyperparam_values==equiv_hyperparam_values):
                    # If all entries are the same, add the equivalent hyperparameter to the list of redundant hyperparameter names
                    redundant_hyperparam_names.append(equiv_hyperparam_name)

                    # Inform the user
                    print(f"The hyperparameters '{key_hyperparam_name}' and '{equiv_hyperparam_name}' have the same values in each run and are therefore equivalent, thus remove the 'redundant' hyperparameter '{equiv_hyperparam_name}' from 'raw_runs_df'.")
                else:
                    # Otherwise, throw an error
                    err_msg = f"The hyperparameters '{key_hyperparam_name}' and '{equiv_hyperparam_name}' do not have the same values in each run and are therefore not equivalent, please correct the input 'equiv_hyperparams_dict'."
                    raise ValueError(err_msg)
        
        # Cast 'redundant_hyperparam_names' to a set of the same name
        redundant_hyperparam_names = set(redundant_hyperparam_names)

        # Differ cases where the 'redundant_cols' is still None after step 1 (which is the case if no redundant columns found in step 1) or not
        if redundant_cols is None:
            # If 'redundant_cols' is still None, assign the redundant hyperparameters to it
            redundant_cols = redundant_hyperparam_names
        else:
            # If 'redundant_cols' is not None (it will be a set), take the set-union of the previously found redundant columns
            # and the columns that correspond to the determined redundant hyperparameters
            redundant_cols = redundant_cols.union(redundant_hyperparam_names)
        ##############################################################################################################################

        ##############################################################################################################################
        # Step 3:
        # Loop over the colum names and try to find any that have different values in each row.
        # In case that hyper parameters are swept the will have the same values over different cross validation folds.
        # Thus any configuration that is different for each row can not be 'real' hyperparameters but must be configurations
        # that are specific to the cross-valiation sets (e.g. train) and somehow depend on a 'real' hyperparamter.
        # An example for this arises if the exclusion range in the binary classification scenario is treated as hyperparameter.
        # This exclusion range can be defined over summary statistics of the train set that will lead to configurations that are
        # different for each 'exclusion range' and also over each cross validation (e.g. lower and upper boundaries of
        # label 0 and label 1, respectively). These configurations (i.e. their associated columns) should be removed.
        ##############################################################################################################################
        # Loop over all columns
        col_with_fully_unique_values = list()
        for col_name in raw_runs_df.columns:
            # For some special columns, it is also expected that they have different entries in all rows.
            # In case the current column is one of these, continue to the next column
            #if col_name in ['run_folder_path', 'valid_metric_values']:
            if col_name in ['run_folder_path', 'train_metric_values', 'valid_metric_values', self.hyperparam_name_map['cv_fold_index']]:
                continue
        
            # Get the column values
            col_values = raw_runs_df[col_name]

            # Get the number of unique column values
            try:
                num_unique_col_values = len(np.unique(col_values))
            except TypeError:
                # In case there is a type error (e.g. in case that there are numeric and string entries) transform
                # the column values to strings before determining the unique
                num_unique_col_values = len(np.unique([str(item) for item in col_values]))

            # Check if the number of unique column values is equal to the number of entries/rows of the raw_runs DataFrame
            # that would mean that all the column entries are different
            if num_unique_col_values==len(raw_runs_df):
                # Append the current column name to the name of columns that contain fully unique values (so all values are different)
                col_with_fully_unique_values.append(col_name)

        # Cast 'col_with_fully_unique_values' to a set of the same name
        col_with_fully_unique_values = set(col_with_fully_unique_values)

        # In case that some columns have been found that contain different values, they are considered to be 'redundant'
        # because they actually depend on a 'real' hyperparameter that will be non-redundant
        if 0<len(col_with_fully_unique_values):
            # In case that redundant_cols is None (so it has not been defined before), assign the aforementioned column names as 
            # set to it
            if redundant_cols is None:
                redundant_cols = col_with_fully_unique_values
            else:
                # If 'redundant_cols' is not None (it will be a set), take the set-union of the previously found redundant columns
                # and the aforementioned column names
                redundant_cols = redundant_cols.union(col_with_fully_unique_values)
        ##############################################################################################################################

        # Remove the redundant columns in 'raw_runs_df' and return the result
        return raw_runs_df.drop(columns=redundant_cols)

    def generate_combined_runs_df(self):
        """
        Combine information of all runs with the same hyperparameter values combination
        but with different cross validation folds.
        
        Remark: Although the cross validation fold index is a hyperparameter, here all but
                this hyperparameter are meant with 'same hyperparameter values combination'!
            
        Args:
            None

        Return:
            (pandas.DataFrame): DataFrame containing the combined information for each 
                hyperparameter values combination.
        """
        print()
        
        # Get all non-redundant hyperparameter names
        hyperparam_names = self.non_redundant_hyperparam_names

        # Remove the hyperparameter names associated with cross validation (cv) if they are non-redundant hyperparameter names
        for cv_hyperparam_name in self.cross_validation_hyperparam_names:
            if cv_hyperparam_name in hyperparam_names:
                hyperparam_names.remove(cv_hyperparam_name)

        # Generate the combined_runs_dict differing cases where hyperparemeters were scanned (hyperparam_names not empty) or not (hyperparam_names is empty)
        combined_runs_dict = collections.defaultdict(list)
        if len(hyperparam_names)==0:
            # Define the hyperparameters dict as empty dictionary here
            hyperparams_dict = dict()

            # Update the combined runs dictionary for the current hyperparameter group
            return_flag = self.update_combined_runs_dict_for_hyperparam_group(hyperparams_dict, self.raw_runs_df, combined_runs_dict)

        else:
            # Group the raw_runs_df DataFrame by all different combinations of these hyperparameters
            for hyperparam_values, group_df in self.raw_runs_df.groupby(hyperparam_names):
                # Generate the hyperparameter dictionary of the current group, while
                # differing the cases where there is only one hyperparameter or more
                if len(hyperparam_names)==1:
                    # In case there is only one hyperparameter, hyperparam_names is an iterable
                    # with only one element and the variable hyperparam_values is a number or string.
                    hyperparams_dict = {hyperparam_names[0]: hyperparam_values}
                else:
                    # For more than one hyperparameter, zip the hyperparameter names and values
                    # (that are BOTH iterables) to generate the dictionary
                    hyperparams_dict = dict( zip(hyperparam_names, hyperparam_values) )

                # Update the combined runs dictionary for the current hyperparameter group
                return_flag = self.update_combined_runs_dict_for_hyperparam_group(hyperparams_dict, group_df, combined_runs_dict)
        
        # Cast the combined_runs dictionary to a DataFrame, drop duplicates and return it
        return pd.DataFrame(combined_runs_dict).drop_duplicates()
    
    def update_combined_runs_dict_for_hyperparam_group(self, 
                                                       hyperparams_dict, 
                                                       group_df, 
                                                       combined_runs_dict):
        """
        Add entries to the 'combined_runs_dict' for the current hyperparameter group.
        
        Args:
            hyperparams_dict (dict): Dictionary corresponding to a map from hyperparameter name to their value.
            group_df (pandas.DataFrame): DataFrame holding the entries for the hyperameters values (listed in hyperparams_dict).
            combined_runs_dict (dict): Dictionary of lists (passed by reference) that is updated by this function.

        Returns:
            None
        
        """
        # Extract the number of cross validation folds the model should have been trained for
        # and throw an error if this number is not unique within the group.
        unique_K = list(set( group_df[self.hyperparam_name_map['num_cv_folds']] ))
        if 1<len(unique_K):
            err_msg = f"The number of cross validation folds (K) is not the same for all runs with hyperparameter values combinations:\n{hyperparams_dict}."
            raise ValueError(err_msg)
        K = unique_K[0]

        # Check that the output for all cross validation folds exists for the current 
        # hyperparameter values combination (=group):
        # 1) Get the set of cross validation fold indices for the current group
        k_set = set(group_df[self.hyperparam_name_map['cv_fold_index']])
        # 2) Check that this set is equivalent to the set {0, 1, ..., K-1}.
        #    If this is not the case, inform the user and continue to the next group
        #    Remark: Because K is unique and the model can not be trained for any k=>K,
        #            we only need to check that all k that all k are present.
        diff_set = set(range(K)) - k_set
        if 0<len(diff_set):
            print(f"ATTENTION: No information present in 'raw_runs_dict' for the cross validation folds '{list(diff_set)}' for the hyperparameter values combination:\n{hyperparams_dict}\n\n")

        ################################################################################################################################################################
        ### Train set epochs and metrics
        ################################################################################################################################################################
        # Determine the unique train epochs (tuples) of the resulting list of train epochs
        unique_train_epochs = list(set( [train_epochs for train_epochs in group_df['train_epochs']] ))
        
        # Check that all train epochs are the same over the group (<=> there is only one unique train epochs entry), 
        # if this is not the case inform the user and continue to the next group
        if len(unique_train_epochs)!=1:
            print(f"The train epochs for which the validation metric was recorded are not equivalent for the cross validation folds of the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Assign the unique train epochs entry to a train epochs variable
        train_epochs = unique_train_epochs[0]

        # Stack the train metric values for the different cross validation (CV) folds 
        # thereby obtaining a 2d numpy array of shape (#CV-folds, #metric_values)
        stacked_train_metric_values = np.vstack(group_df['train_metric_values'])

        # In case that any train metric value is np.nan, inform the user and continue to the next group
        if np.any(np.isnan(stacked_train_metric_values)):
            print(f"At least one train metric value was NaN for the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Determine certain summary statistics over the folds (first axis) for each recorded train epoch value (second axis)
        # and cast the resutling 1d numpy arrays to tuples
        mean_train_metric_values = tuple( np.mean(stacked_train_metric_values, axis=0) )
        std_train_metric_values  = tuple( np.std(stacked_train_metric_values, axis=0) )
        min_train_metric_values  = tuple( np.min(stacked_train_metric_values, axis=0) )
        max_train_metric_values  = tuple( np.max(stacked_train_metric_values, axis=0) )

        ################################################################################################################################################################
        ### Validation set epochs and metrics
        ################################################################################################################################################################
        # Determine the unique validation epochs (tuples) of the resulting list of validation epochs
        unique_valid_epochs = list(set( [valid_epochs for valid_epochs in group_df['valid_epochs']] ))
        
        # Check that all validation epochs are the same over the group (<=> there is only one unique validationepochs entry), 
        # if this is not the case inform the user and continue to the next group
        if len(unique_valid_epochs)!=1:
            print(f"The validation epochs for which the validation metric was recorded are not equivalent for the cross validation folds of the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Assign the unique valid epochs entry to a validation epochs variable
        valid_epochs = unique_valid_epochs[0]

        # Stack the validation metric values for the different cross validation (CV) folds 
        # thereby obtaining a 2d numpy array of shape (#CV-folds, #metric_values)
        stacked_valid_metric_values = np.vstack(group_df['valid_metric_values'])

        # In case that any validation metric value is np.nan, inform the user and continue to the next group
        if np.any(np.isnan(stacked_valid_metric_values)):
            print(f"At least one validation metric value was NaN for the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Determine certain summary statistics over the folds (first axis) for each recorded validation epoch value (second axis)
        # and cast the resutling 1d numpy arrays to tuples
        mean_valid_metric_values = tuple( np.mean(stacked_valid_metric_values, axis=0) )
        std_valid_metric_values  = tuple( np.std(stacked_valid_metric_values, axis=0) )
        min_valid_metric_values  = tuple( np.min(stacked_valid_metric_values, axis=0) )
        max_valid_metric_values  = tuple( np.max(stacked_valid_metric_values, axis=0) )

        ################################################################################################################################################################
        ### Optimal validation set epochs and metrics
        ################################################################################################################################################################
        # Get the optimal metric value and associated epoch as well as metric error
        # Remark: As value use the mean and as error use the standard deviation 
        #         over the values of the different fold.
        optimal_valid_epoch, optimal_valid_metric_value, optimal_valid_metric_error = self.get_optimal_metric_value(valid_epochs, mean_valid_metric_values, std_valid_metric_values)

        ################################################################################################################################################################
        ### Learning rate
        ################################################################################################################################################################
        # Determine the unique learning rate epochs (tuples) of the resulting list of learning rate epochs
        unique_learning_rate_epochs = list(set( [learning_rate_epochs for learning_rate_epochs in group_df['learning_rate_epochs']] ))
        
        # Check that all learning rate epochs are the same over the group (<=> there is only one unique learning rate epochs entry), 
        # if this is not the case inform the user and continue to the next group
        if len(unique_learning_rate_epochs)!=1:
            print(f"The learning rate epochs for which the learning rate was recorded are not equivalent for the cross validation folds of the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Assign the unique learning rate epochs entry to a learning rate epochs variable
        learning_rate_epochs = unique_learning_rate_epochs[0]

        # Determine the unique learning rate values (tuples) of the resulting list of learning rate values
        unique_learning_rate_values = list(set( [learning_rate_values for learning_rate_values in group_df['learning_rate_values']] ))
        
        # Check that all learning rate values are the same over the group (<=> there is only one unique learning rate values entry), 
        # if this is not the case inform the user and continue to the next group
        if len(unique_learning_rate_values)!=1:
            print(f"The learning rate values for which the learning rate was recorded are not equivalent for the cross validation folds of the hyperparameter values combination:\n{hyperparams_dict}.\n =>Skip this combination in the generatation of 'combined_runs_df'!\n")
            return

        # Assign the unique learning rate values entry to a learning rate values variable
        learning_rate_values = unique_learning_rate_values[0]

        ################################################################################################################################################################
        ### Add values to the combined runs DataFrame
        ################################################################################################################################################################
        # Append the hyperparameters to the combined_runs dictionary
        for hyperparam_name, hyperparam_value in hyperparams_dict.items():
            combined_runs_dict[hyperparam_name].append(hyperparam_value)

        # Append the epochs, summary statistics of the metric values for validation and train and the optimal validation values (epoch, metric value, and metric error) 
        # to the corresponding lists
        # Optimal validation metric values
        combined_runs_dict['optimal_valid_epoch'].append(optimal_valid_epoch)
        combined_runs_dict['optimal_valid_metric_value'].append(optimal_valid_metric_value)
        combined_runs_dict['optimal_valid_metric_error'].append(optimal_valid_metric_error)

        # Validation matric values
        combined_runs_dict['valid_epochs'].append(valid_epochs)
        combined_runs_dict['mean_valid_metric_values'].append(mean_valid_metric_values)
        combined_runs_dict['std_valid_metric_values'].append(std_valid_metric_values)
        combined_runs_dict['min_valid_metric_values'].append(min_valid_metric_values)
        combined_runs_dict['max_valid_metric_values'].append(max_valid_metric_values)

        # Train metric values
        combined_runs_dict['train_epochs'].append(train_epochs)
        combined_runs_dict['mean_train_metric_values'].append(mean_train_metric_values)
        combined_runs_dict['std_train_metric_values'].append(std_train_metric_values)
        combined_runs_dict['min_train_metric_values'].append(min_train_metric_values)
        combined_runs_dict['max_train_metric_values'].append(max_train_metric_values)

        # Learning rate values
        combined_runs_dict['learning_rate_epochs'].append(learning_rate_epochs)
        combined_runs_dict['learning_rate_values'].append(learning_rate_values)

    def get_optimal_metric_value(self, 
                                 epochs, 
                                 metric_values, 
                                 metric_errors):
        """ 
        Determine the optimal metric value (and corresponding epoch and metric error) for the passed metric values.

        Args:
            epochs (iterable): Iterable (e.g. tuple) containing the epochs.
            metric_values (iterable): Iterable (e.g. tuple) containing the metric values for each epoch entry.
            metric_errors (iterable): Iterable (e.g. tuple) containing the metric error for each epoch entry.

        Returns:
            (int, float, float): Optimal epoch, optimal metric value, and optimal metric error.

        """
        # Differ cases where the optimal metric value should either be determined 
        # over all epochs or taken as the metric value of the last epoch
        if self.optimal_metric_value=='over_all_epochs':
            # Find the index of the optimal (i.e., minimal or maximal) (metric) value over all epochs
            if self.metric_optimization=='min':
                opt_ind = np.argmin(metric_values)
            elif self.metric_optimization=='max':
                opt_ind = np.argmax(metric_values)
        elif self.optimal_metric_value=='last_epoch':
            # If the optimal metric value should correspond to the metric value of
            # the last epoch, the opt_ind is given by the index of the last epoch
            opt_ind = len(epochs)-1

        # Return the (minimizing) epoch, minimal metric value, and 0 (this method has no error estimation)
        return epochs[opt_ind], metric_values[opt_ind], metric_errors[opt_ind]


    def determine_globally_optimal_args_dict(self):
        """
        Determine the globally optimal hyperparameters as dictionary.

        Arg:
            None

        Return:
            (dict, float): Dictionary containing the hyperparameter names for which the validation metric
                is globally optimal (over all hyperparameter settings) as first return variable.
                Optimal metric value as second return value.

        """
        # Determine the globally optimal validation metric value differing the minimization and maximization cases
        if self.metric_optimization=='min':
            global_optimal_valid_metric_value = self.combined_runs_df['optimal_valid_metric_value'].min()
        elif self.metric_optimization=='max':
            global_optimal_valid_metric_value = self.combined_run_df['optimal_valid_metric_value'].max()

        # Get the row entries of the combined_runs DataFrame of the row that contains optimal validation metric value and make it a dictionary
        # Remark: 'x.to_dict('records')[0]' is a trick to map any series object 'x' to a dictionary
        optimal_row_dict = self.combined_runs_df[self.combined_runs_df['optimal_valid_metric_value']==global_optimal_valid_metric_value].to_dict('records')[0]

        # Loop over this dictionary of the optimal row (its entries belong to different columns) 
        # and construct the optimal arguments (for hydra) dictionary
        optimal_args_dict = dict()
        for col_name, col_value in optimal_row_dict.items():
            # If the current column name is 'optimal_epoch', use this as value for hyperparameter
            # name corresponding to the number of epochs
            if col_name=='optimal_epoch':
                optimal_args_dict[self.hyperparam_name_map['num_epochs']] = col_value
            else:
                # Otherwise, differ cases where the column name is a hyperparameter name or not
                if col_name in self.hyperparam_names_set:
                    # If the column name is a hyperparameter name, add a key-value pair with 
                    # the column name and value to the optimal arguments dictionary
                    optimal_args_dict[col_name] = col_value
                else:
                    # Otherwise, continue to the next column
                    continue
                
        # Return this dictionary and the globally optimal validation metric value
        return optimal_args_dict, global_optimal_valid_metric_value

    def generate_missing_args_dict_list(self):
        """
        Generate a list containing the missing arguments dictionaries.

        Args:
            None
        
        Return:
            (list of dict): List of arguments dictionaries that each have the
                form {<arg_name>:<arg_value} for different arguments.
        
        """
        # Loop over the problematic runs attribute (that is a list whose elements are run folder paths of the problematic runs)
        missing_args_dict_list = list()
        self.missing_runs      = list()
        for run_folder_path in self.problematic_runs:
            # Get the flattened config dictionary for the current run
            # Remark: The method 'get_flattened_config_dict' returns None in case no config file exists in the run folder.
            #         If this is the case, continue to the next run
            flattened_config_dict = self.get_flattened_config_dict(run_folder_path)
            if flattened_config_dict is None:
                print(f"No config dictionary has been found for the problematic run in {run_folder_path}\n")
                continue

            # Construct a dictionary that contains all key-value-pairs of the flattened config 
            # dictionary whose key corresponds to the non-redundant hyperparameter names
            args_dict = {key: value for key, value in flattened_config_dict.items() if key in self.non_redundant_hyperparam_names}

            # If there is no row in self.raw_runs_df that contains the argument combination
            # specified in the args dict, add the args dict and the run folder path to their
            # corresponding lists of missing arguments.
            # Remark: If no such row exists it means that the current arguments combination of
            #         the problematic run is actually missing in the raw runs dictionary.
            #         Thus this current arguments combination is considered 'missing'.
            if not self.is_row_with_values_in_df(args_dict, self.raw_runs_df):
                missing_args_dict_list.append(args_dict)
                self.missing_runs.append(run_folder_path)

        # Remove duplicate elements in the list of missing arguments combinations
        # First, transform the list of dictionaries of the form 
        # {arg_name_1: arg_value_1, arg_name_2: arg_value_2, ...} 
        # to a list of tuples of 2-tuple pairs of the form 
        # ((arg_name_1, arg_value_1), (arg_name_2, arg_value_2), ...)
        # and cast it to a set (thereby removing duplicates).
        # Remark: The dictionaries are not hashable and thus have to be mapped
        #         to the tuples (of 2-tuple pairs) that are hashable, which is
        #         required to obtain a set
        missing_args_tuple_set = {tuple(missing_args_dict.items()) for missing_args_dict in missing_args_dict_list}
        
        # Second, reconstruct the dictionaries from the tuples of 2-tuples in
        # in the generated set and return it
        return [dict(item) for item in set(missing_args_tuple_set)]

    def is_row_with_values_in_df(self, 
                                 dict_of_values, 
                                 df):
        """
        Check if there exists a row in the DataFrame 'df' that 
        has the values specified in 'dict_of_values'.

        Args:
            dict_of_values (dict): Dictionary of values whose keys should
                correspond to column names of the DataFrame 'df'.
            df (pandas.DataFrame): DataFrame for which the existence of
                a row with the values specified in 'dict_of_values' should
                be checked.
        
        Return:
            (bool): Boolean flag specifiying if such a row exists or not in 'df'.
        
        """
        # Generate boolean series object (a mask) with an entry for each of the 
        # rows of 'df' and all elements initially set to True
        mask_series = df.iloc[:, 0] == df.iloc[:, 0]
        
        # Loop over the key-value pairs in the dictionary 'dict_of_values'
        # and update the mask series with the boolean series containing
        # True for each row where the dictionary-value matches the column value
        # of 'df' for the column corresponding to the dictionary-key.
        # All other values with inequality will be False.
        for key, value in dict_of_values.items():
            mask_series &= (df[key] == value)
        
        # Check if there is any True value in the mask series which would correspond
        # to a row that contains the values specified in 'dict_of_values'
        return mask_series.any()

    def create_args_combinations_file_for_missing_runs(self):
        """ Generate an arguments combinations file with the intent to repeat the missing runs. """
        # Create the arguments combinations file passing the class attribute corresponding to
        # the missing arguments dictionary list and using 'missing_runs' as the file name for 
        # the to be created file
        self.create_args_combinations_file(self.missing_args_dict_list, 'missing_runs')

    def create_args_combinations_file(self, 
                                      args_dict_list, 
                                      file_name):
        """
        Create an arguments combinations file that can be read by 'run_jobs.py' and
        used to run jobs for each line (corresponding to one specific arguments combination).

        Args:
            args_dict_list (list of dict): List of arguments dictionaries that each have the
                form {<arg_name>:<arg_value} for different arguments.
            file_name (str): Name that should be used for the created file.
        """
        # Loop over the argument dictionaries in args_dict_list
        args_combinations = list()
        for args_dict in args_dict_list:
            # Sort the arguments by their name (keys of the arguments dictionary)
            args_names = list(args_dict.keys())
            args_names.sort()

            # Generate the arguments combination for the current arguments dictionary
            # that corresponds to a list containing argument name-value pairs of the form {arg_name}={arg_value}
            args_combination = [self.stringify_arg_name_value_pair(arg_name, args_dict[arg_name]) for arg_name in args_names]

            # Append the arguments combination to the list of arguments combinations
            args_combinations.append(args_combination)

        # Call the utils function 'create_args_combinations_file' that creates the arguments
        # combinations file for an arguments combinations input that is a list of 'arguments
        # combination' that are themselves lists containing argument name-value combinations 
        # in the form: f'"{arg_name}={arg_value}"'
        # The returned variable of this function is the path to the created file
        args_comb_file_path = self._create_args_combinations_file(args_combinations, 
                                                                  file_name, 
                                                                  args_combinations_files_dir_path=self.args_combinations_files_dir_path)

        # Inform the user what the path to the created file is
        print(f"Created the arguments combinations file: {args_comb_file_path}")

    def stringify_arg_name_value_pair(self, 
                                      arg_name, 
                                      arg_value):
        """ 
        Stringify an argument name-value pair to be used as override argument (command line) for hydra. 

        Args:
            arg_name (str): Argument name.
            arg_value (int, float, or str): Argument value.
        
        Return:
            (str): Stringified argument name-value pair.
        
        """
        return f"{arg_name}={arg_value}"
    
    def _create_args_combinations_file(self, 
                                       args_combinations, 
                                       file_name, 
                                       args_combinations_files_dir_path='./job_management/args_combinations'):
        """
        Create an argument combinations file bases on the passed arguments combinations.
    
        Args:
            args_combinations (list): List of 'arguments combination' that are themselves lists
                containing argument name-value combinations in the form: f'"{arg_name}={arg_value}"'
                Ideally, these lists corresponding to an arguments combination are sorted by the 
                argument name for reproducibility.
            file_name (str): Name of the to be created argument combinations file.
            args_combinations_files_dir_path (str or Path): Path to the argument combination files.
                (Default: './job_management/args_combinations')
            
        Return:
            (str): Path to the created arguments combinations file as string.
    
        Remark: The created argument combinations file has one 'argument combinations' per line in the form:
                "<arg_1_name>=<arg_1_value>" "<arg_2_name>=<arg_2_value>" "<arg_3_name>=<arg_3_value>"
        
        """
        # Open a new writable text file
        args_comb_file_path = str( Path(args_combinations_files_dir_path, f"{file_name}.txt") )
        with open(args_comb_file_path, 'w') as f:
            # Loop over all arguments combinations and save each of them as new line to the text file
            for index, args_combination in enumerate( args_combinations ):
                # Separate the arguments using a single white space
                args_combination_str = " ".join(args_combination)
    
                # Write a new line with this string
                f.write(args_combination_str + '\n')
    
        # Return the path to the arguments combinations file
        return args_comb_file_path

    def plot_metric_curves(self, 
                           y_min=None, 
                           metric_label='Metric values', 
                           fixed_hyperparams_dict={}):
        """ 
        Plot the metric curves. 

        Args:
             y_min (float or None): Minimal y-value to be used as the lower y-axis limit
                as float or None if the limit should be chosen by matplotlib.pyplot.
                (Default: None)
            metric_label (str): Label to be used for y-axis (showing the metric values).
                (Default: 'Metric values')
            fixed_hyperparams_dict (dict): Dictionary containing hyperaparameters 
                (=dictionary-keys) and to which value they should be fixed 
                (=dictionary-value).
                (Default: {}).
        
        """
        # Get all non-redundant hyperparameter names
        hyperparam_names = self.non_redundant_hyperparam_names

        # Remove the hyperparameter names associated with cross validation (cv) if they are non-redundant hyperparameter names
        for cv_hyperparam_name in self.cross_validation_hyperparam_names:
            if cv_hyperparam_name in hyperparam_names:
                hyperparam_names.remove(cv_hyperparam_name)

        # Differ cases depending on the number of hyperparameter names
        if len(hyperparam_names)==0:
            # Define the hyperparameters dict as empty dictionary here
            hyperparams_dict = dict()

            # Generate a title for the metric curve
            title = "\n".join([f"{hyperparam_name}={hyperparam_value}" for hyperparam_name, hyperparam_value in hyperparams_dict.items()])

            # Plot the metric curve for the entire combined_runs_df
            self._plot_metric_curve_for_data_frame(self.combined_runs_df, y_min=y_min, title=None, metric_label=metric_label)

        else:
            # Group the combined_runs_df DataFrame by all different combinations of these hyperparameters
            for hyperparam_values, group_df in self.combined_runs_df.groupby(hyperparam_names):
                # Generate the hyperparameter dictionary of the current group, while
                # differing the cases where there is only one hyperparameter or more
                if len(hyperparam_names)==1:
                    # In case there is only one hyperparameter, hyperparam_names is an iterable
                    # with only one element and the variable hyperparam_values is a number or string.
                    hyperparams_dict = {hyperparam_names[0]: hyperparam_values}
                else:
                    # For more than one hyperparameter, zip the hyperparameter names and values
                    # (that are BOTH iterables) to generate the dictionary
                    hyperparams_dict = dict( zip(hyperparam_names, hyperparam_values) )

                # Loop over all current hyperparameter values and check if they all correspond to their fixed (i.e. requested) values
                # In case that any current value doesn't correspond to the fixed value, skip the current hyperparameter combination
                # by continuing to the next combination
                skip_current_combination = False
                for hyperparam_name, hyperparam_value in hyperparams_dict.items():
                    if hyperparam_name in fixed_hyperparams_dict:
                        if hyperparam_value!=fixed_hyperparams_dict[hyperparam_name]:
                            skip_current_combination = True

                if skip_current_combination:
                    continue

                # Generate a title for the metric curve
                title = "\n".join([f"{hyperparam_name}={hyperparam_value}" for hyperparam_name, hyperparam_value in hyperparams_dict.items()])

                # Plot the metric curve for the current group DataFrame
                self._plot_metric_curve_for_data_frame(group_df, y_min=y_min, title=title, metric_label=metric_label)
                

    def _plot_metric_curve_for_data_frame(self, 
                                          input_df, 
                                          y_min=None, 
                                          title=None, 
                                          alpha=0.5, 
                                          metric_label='Metric values'):
        """ 
        Plot the metric curve for rows of a data frame. 

        Args:
            input_df (pandas.DataFrame): The input DataFrame holding the metric-curve information 
                as entries that should be plotted.
            y_min (float or None): Minimal y-value to be used as the lower y-axis limit
                as float or None if the limit should be chosen by matplotlib.pyplot.
                (Default: None)
            title (str or None): Title to be used or None if no title should be set.
                (Default: None)
            alpha (float): Transparency to be used in the plot.
                (Default: 0.6)
            metric_label (str): Label to be used for y-axis (showing the metric values).
                (Default: 'Metric values')
        
        """
        ################################################################################################################
        ### Preparation: Access the epochs and corresponding values
        ################################################################################################################
        # Train set
        train_epochs             = np.array(list(input_df['train_epochs'])).squeeze()
        mean_train_metric_values = np.array(list(input_df['mean_train_metric_values'])).squeeze()
        min_train_metric_values  = np.array(list(input_df['min_train_metric_values'])).squeeze()
        max_train_metric_values  = np.array(list(input_df['max_train_metric_values'])).squeeze()

        # Validation set
        valid_epochs             = np.array(list(input_df['valid_epochs'])).squeeze()
        mean_valid_metric_values = np.array(list(input_df['mean_valid_metric_values'])).squeeze()
        min_valid_metric_values  = np.array(list(input_df['min_valid_metric_values'])).squeeze()
        max_valid_metric_values  = np.array(list(input_df['max_valid_metric_values'])).squeeze()

        # Learning rate
        learning_rate_epochs = np.array(list(input_df['learning_rate_epochs'])).squeeze()
        learning_rate_values = np.array(list(input_df['learning_rate_values'])).squeeze()

        ################################################################################################################
        ### Make the figure
        ################################################################################################################
        ### Preparations
        # Define some plot specs
        ax2_color = 'orange'

        # Make the figure
        plt.figure()

        # Set a title
        if title is not None:
            plt.title(title)

        # Get the current (primary) axis
        ax1 = plt.gca()

        # Make a secondary axis
        # (twin object for two different y-axis on the sample plot)    
        ax2 = ax1.twinx()

        ### Secondary axis plots
        # Plot the learning rate
        ax2.plot(learning_rate_epochs, learning_rate_values, color=ax2_color, ls='--', lw=2, label='Learning rate')

        ### Primary axis plots
        # Plot the training metric
        ax1.fill_between(train_epochs, y1=min_train_metric_values, y2=max_train_metric_values, color='b', label='train (min-max)', alpha=alpha)
        ax1.plot(train_epochs, mean_train_metric_values, color='b', ls='-', label='train (mean)')

        # Plot the validation metric
        ax1.fill_between(valid_epochs, y1=min_valid_metric_values, y2=max_valid_metric_values, color='r', label='valid (min-max)', alpha=alpha)
        ax1.plot(valid_epochs, mean_valid_metric_values, color='r', ls='-', label='valid (mean)')

        ### Plot specs
        # Set plot specs for primary axis
        ax1.set_ylabel(metric_label)
        if y_min is not None:
            ax1.set_ylim([y_min, None])
        ax1.legend()

        # Set plot specs for primary axis
        ax2.set_ylabel('Learning rate')
        # Convert y-axis to logarithmic scale
        ax2.set_yscale("log")
        # Set colors of second axis
        ax2.yaxis.label.set_color(ax2_color)
        ax2.tick_params(axis='y', colors=ax2_color)
        ax2.spines['right'].set_color(ax2_color)

        # Set global plot specs (for both axes)
        plt.xlabel('Epochs')
        plt.show()
       

    def plot_metric_curve_for_run(self, 
                                  run_folder_path, 
                                  y_min=None, 
                                  title=None):
        """ 
        Plot the metric curve of a specific run. 
        
        Args:
            run_folder_path (str or Path): Path to the run folder.
            y_min (float or None): Minimal y-value to be used as the lower y-axis limit
                as float or None if the limit should be chosen by matplotlib.pyplot.
                (Default: None)
            title (str or None): Title to be used or None if no title should be set.
                (Default: None)
        
        """
        # Check that the run folder path exists
        if not os.path.isdir(run_folder_path):
            err_msg = f"No directory exists in {run_folder_path}"
            raise FileNotFoundError(err_msg)

        # Get a dictionary containing the tracked quantities (e.g. training loss, train metric, validation metric)
        # over several epochs for the current run.
        # Remark: The method 'get_tracked_quantities_dict' returns None in case no tracked quantities file exists in the run folder.
        #         If this is the case throw an error.
        tracked_quantities_dict = self.get_tracked_quantities_dict(run_folder_path)
        if tracked_quantities_dict is None:
            err_msg = f"The 'tracked_quantites' file does not exists in {run_folder_path}"
            raise FileNotFoundError(err_msg)

        # Get the epochs and corresponding metric values for the validation and training sets
        train_epochs, train_metric_values = self.get_epochs_and_metric_values(tracked_quantities_dict, set_name='train')
        valid_epochs, valid_metric_values = self.get_epochs_and_metric_values(tracked_quantities_dict, set_name='valid')

        # Plot the metric curve
        plt.figure()
        if title is not None:
            plt.title(title)
        plt.plot(train_epochs, train_metric_values, 'b-', label='train')
        plt.plot(valid_epochs, valid_metric_values, 'r-', label='valid')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        if y_min is not None:
            plt.ylim([y_min, None])
        plt.legend()
        plt.show()


    def plot_validation_metric_values(self, 
                                      fixed_hyperparams_dict={}, 
                                      display_global_optimum=True, 
                                      display_errors=False, 
                                      num_sp_per_row=3, 
                                      show_delta=False, 
                                      metric_label='Metric values'):
        """
        Plot the optimal metric values for different (non-redundant) hyperparameters.

        Args:
            fixed_hyperparams_dict (dict): Dictionary containing to be fixed hyperparameter names 
                as dictionary-keys and the values (they should be fixed to) as dictionary-values.
                (Default: {} -> Do not fix any configuration parameters)
            display_global_optimum (bool): Should the global optimum be indicated by a vertical line
                in each configuration parameter panel?
                (Default: True)
            display_errors (bool): Should validation metric errors be displayed?
                (Default: False)
            num_sp_per_row (int): Specify how many subplots (sp) should be displayed per row.
                (Default: 4)
            show_delta (bool): Boolean flag if the 'delta values' (mean difference between adjacent hyperparameter values)
                should be shown in the plot or not.
                (Default: True)
            metric_label (str): Label to be used for y-axis (showing the metric values).
                (Default: 'Metric values')

        Return:
            None
            
        """
        # Filter out all dataframe entries that do not contain the fixed hyperparameter values:
        # 1) Loop over the fixed hyperparameters and then generate an index array for the rows
        #    that should be kept in step (2)
        inds = np.ones(len(self.combined_runs_df), dtype=bool)
        for hyperparam_name, fixed_hyperparam_value in fixed_hyperparams_dict.items():
            inds &= (self.combined_runs_df[hyperparam_name]==fixed_hyperparam_value)

        # 2) Filter out any hyperparameter values combinations that do not contain the fixed 
        #    hyperparameter values using the indices determined in (1)
        filtered_combined_runs_df = self.combined_runs_df[inds]
        
        # TODO: Prettify the following code:
        # The hyperparameters to be plotted are all non-redundant hyperparameters without the
        # hyperparameters associated with cross validation and without the fixed hyperparameters
        # Remark: Sort the to be plotted hyperparameter names
        plot_hyperparam_names = set(self.non_redundant_hyperparam_names) - set(self.cross_validation_hyperparam_names) - set(fixed_hyperparams_dict.keys())
        plot_hyperparam_names = list(plot_hyperparam_names)
        plot_hyperparam_names.sort()

        # Make a multipanel figure
        # Remark: First calculate how many rows and columns the figure should have.
        #         The number of rows is given by the ceiled ratio of total number of subplots (sp)
        #         (corresponding to the number of to be ploted hyperparameters) and the number
        #         of panels per row.
        #         The number of columns is corresponds to the number of subplots (sp) if there are 
        #         more hyperparamters to be plotted than fit in one row or else corresponds to the
        #         number of hyperparameters
        num_rows = int(np.ceil( len(plot_hyperparam_names)/num_sp_per_row ))
        num_cols = num_sp_per_row if num_sp_per_row<=len(plot_hyperparam_names) else len(plot_hyperparam_names)
        figsize  = [num_cols*7, num_rows*6.5]
        fig, axs = plt.subplots(num_rows, num_cols, squeeze=False, figsize=figsize)

        # Switch the axes lines off for all axis objects
        # Remark: If an axis object will be used for a hyperparameter, the axes lines will be switched on for this
        #         specific axis object. However, it might be that some axis objects will not be associated with any
        #         hyperparameter and in this case the axes lines of these axis objects will stay turned off.
        for index_0 in range(len(axs)):
            for index_1 in range(len(axs[index_0])):
                axs[index_0, index_1].axis('off')

        # Make a title for the whole figure in case that 'fixed_hyperparams_dict' is not empty
        if len(fixed_hyperparams_dict)>0:
            fig.suptitle(f"fixed={fixed_hyperparams_dict}")

        # Loop over all to be plotted hyperparameters
        for hyperparam_index, hyperparam_name in enumerate(plot_hyperparam_names):
            # Calculate the subplot indices
            sp_index_x = hyperparam_index%num_cols
            sp_index_y = int(np.floor( hyperparam_index/num_cols))

            # Get the corresponding axis (object) for the current subplot
            ax = axs[sp_index_y, sp_index_x]

            # Switch the axes lines on for the current axis object
            ax.axis('on')

            # Get all plot hyperparameter but the current hyperparameter
            other_hyperparam_names = copy.deepcopy(list(plot_hyperparam_names))
            other_hyperparam_names.remove(hyperparam_name)

            # Differ cases if there are other hyperparameters or not
            valid_metric_values_dict = collections.defaultdict(list)
            if len(other_hyperparam_names)==0:
                # Plot the 'metric curve' for the current hyperparameter (there are no other hyperparameters) 
                # and update 'valid_metric_values_dict' with the plotted values
                valid_metric_values_dict = self._plot_validation_metric_values(ax, hyperparam_name, filtered_combined_runs_df, valid_metric_values_dict)

            else:
                # In case there are other hyperparameters, group the filtered dataframe by the different combinations of the other 
                # hyperparameter values and loop over these groups
                for group_index, (other_hyperparam_values, group_df) in enumerate( filtered_combined_runs_df.groupby(other_hyperparam_names) ):
                    # Plot the 'metric curve' for the current hyperparameter and for the current group (unique combination of all
                    # the other hyperparameters) and update 'valid_metric_values_dict' with the plotted values
                    valid_metric_values_dict = self._plot_validation_metric_values(ax, hyperparam_name, group_df, valid_metric_values_dict, color_index=group_index)
            
            # Set the plot specs for the current subplot
            y_min, y_max = self._set_plot_specs_for_plot_validation_metric_values(ax, filtered_combined_runs_df, hyperparam_name, hyperparam_index, sp_index_x, display_global_optimum, metric_label=metric_label)

            # Determine the y-value for the label
            if self.metric_optimization=='min':
                y_label = y_min+(y_max-y_min)*0.05
            elif self.metric_optimization=='max':
                y_label = y_max-(y_max-y_min)*0.05

            # Loop over neighboring hyperparameter values (that are dictionary-keys of valid_metric_values_dict)
            sorted_hyperparam_values = list(valid_metric_values_dict.keys())
            sorted_hyperparam_values.sort()
            for hyperparam_value_1, hyperparam_value_2 in zip(sorted_hyperparam_values[:-1], sorted_hyperparam_values[1:]):
                # Get the validation metric values for each of the hyperparameters
                valid_metric_values_1 = np.array( valid_metric_values_dict[hyperparam_value_1] )
                valid_metric_values_2 = np.array( valid_metric_values_dict[hyperparam_value_2] )

                # In case that the validation metric value arrays of the two hyperparameter do not
                # have the same length, skip the current hyperparameters pair and continue
                # to the next pair
                if len(valid_metric_values_1)!=len(valid_metric_values_2):
                    continue

                # Should the 'delta values be shown'?
                if show_delta:
                    # Calculate the average difference (2)-(1)
                    mean_diff = np.mean( valid_metric_values_2-valid_metric_values_1 )
                    glob_mean = ( np.mean(valid_metric_values_2)+np.mean(valid_metric_values_2) )/2
                    delta     = mean_diff/glob_mean*100

                    # Add the text
                    ax.text((hyperparam_value_1+hyperparam_value_2)/2, y_label, f"{delta: .2f}%", color='k', horizontalalignment='center', verticalalignment='center')

        # Set layout options for the subplots
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

    def _plot_validation_metric_values(self, 
                                       ax, 
                                       hyperparam_name, 
                                       df, 
                                       valid_metric_values_dict, 
                                       color_index=0):
        """
        Plot the validation metric values (along y) as function of the values of a specific hyperparameter 
        (along x) for a unique combination of all the other hyperparameters.
        This plot is called here 'validation metric curve'.

        Args:
            ax (matplotlib axis): Axis to which the plot should be added to.
            hyperparam_name (str): Name of the specific hyperparameter for which the validation metric values
                should be plotted for.
            df (pandas.DataFrame): DataFrame holding the information for a specific hyperparameter filtered
                on a unique combination of all other hyperparameters.
            valid_metric_values_dict (dict): Dictionary containing the values of a specific hyperparameter 
                as dictionary-keys and the associated validation metric values (each for another unique 
                combination of the other hyperparameters) as dictionary-values.
            color_index (int): Color index used to set the color of the curve via f"C{color_index}".
                (Default: 0)

        Return:
            (dict): The updated 'valid_metric_values_dict', where validation metric values have been appended
                to their corresponding hyperparameter values.

        Remark: Because the argument 'valid_metric_values_dict' is a dictionary that is passed by reference,
                it would not be necessary to return it, but this is done to make the code more explicit.

        """
        # Extract the values of the passed data frame
        hyperparam_values   = df[hyperparam_name].to_numpy()
        valid_metric_values = df['optimal_valid_metric_value'].to_numpy()

        # Plot the validation metric values vs. the hyperparameters
        # First plot a line between the points
        ax.plot(hyperparam_values, valid_metric_values, '-', color=f"C{color_index}", lw=2, alpha=0.2, zorder=-1)
        # Second plot a marker for each point
        ax.plot(hyperparam_values, valid_metric_values, 'o', color=f"C{color_index}", ms=5, alpha=0.8)

        # Append each validation metric value to the list of the corresponding hyperparameter value
        for hyperparam_value, valid_metric_value in zip(hyperparam_values, valid_metric_values):
            valid_metric_values_dict[hyperparam_value].append(valid_metric_value)

        return valid_metric_values_dict

    def _set_plot_specs_for_plot_validation_metric_values(self, 
                                                          ax, 
                                                          filtered_combined_runs_df, 
                                                          hyperparam_name, 
                                                          hyperparam_index, 
                                                          sp_index_x, 
                                                          display_global_optimum, 
                                                          axis_fs=12.5, 
                                                          metric_label='Metric values'):
        """ 
        Set the plot specs for the 'plot_validation_metric_values' method. 
        
        Args:
            ax (matplotlib axis): Axis to which the plot should be added to.
            filtered_combined_runs_df (pandas.DataFrame): Pandas DataFrame holding the filtered
                combined run entries.
            hyperparam_name (str): Name of the specific hyperparameter for which the validation metric values
                should be plotted for.
            hyperparam_name (str): Index of the hyperparameter (corresponding to simple enumeration).
            sp_index_x (int): Subplot-index along x (i.e., the column index).
            display_global_optimum (bool): Boolean flag indicating if the global optimum should be displayed
                or not.
            axis_fs (float): Axis fontsize.
                (Default: 12.5)
            metric_label (str): Label to be used for y-axis (showing the metric values).
                (Default: 'Metric values')

        Return:
            (float, float): Min and max y-values.
        
        """
        # Determine if the values of the hyperparameter are numeric or not (in which case they will be strings)
        if np.any([ isinstance(item, str) for item in filtered_combined_runs_df[hyperparam_name] ]):
            are_hyperparam_values_numeric = False
        else:
            are_hyperparam_values_numeric = True
            
        # Determine the minimal and maximal validation metric values within the fixed set ('fixed_') of hyperparameters
        # and determine the hyperparameter value for which the validation metric value is optimal (differing the 
        # cases for metric maximization and and minimization)
        fixed_min_valid_metric_value = filtered_combined_runs_df['optimal_valid_metric_value'].min()
        fixed_max_valid_metric_value = filtered_combined_runs_df['optimal_valid_metric_value'].max()
        if self.metric_optimization=='min':
            fixed_optimal_hyperparam_value = filtered_combined_runs_df[filtered_combined_runs_df['optimal_valid_metric_value']==fixed_min_valid_metric_value][hyperparam_name].iloc[0]
        elif self.metric_optimization=='max':
            fixed_optimal_hyperparam_value = filtered_combined_runs_df[filtered_combined_runs_df['optimal_valid_metric_value']==fixed_max_valid_metric_value][hyperparam_name].iloc[0]

        # Determine the optimal validation metric value in case the global optimum should be displayed
        if display_global_optimum:
            # Remark: The first return value of the method 'determine_globally_optimal_args_dict' is the
            #         arguments dictionary in which we are not interested in here.
            _, global_opt_valid_metric_value = self.determine_globally_optimal_args_dict()

        ##############################################################################################################################################
        # Determine the axis limits
        ##############################################################################################################################################
        if are_hyperparam_values_numeric:
            # Differ the cases where the hyperparameter values should be displayed with a logarithmic (log10) axis or not
            min_hyperparam_value = np.min(filtered_combined_runs_df[hyperparam_name])
            max_hyperparam_value = np.max(filtered_combined_runs_df[hyperparam_name])
            if hyperparam_name in self.log10_hyperparams:
                x_min = 10**( np.log10(min_hyperparam_value)-( np.log10(max_hyperparam_value)-np.log10(min_hyperparam_value) )*0.1 )
                x_max = 10**( np.log10(max_hyperparam_value)+( np.log10(max_hyperparam_value)-np.log10(min_hyperparam_value) )*0.1 )
            else:
                # Differ cases where the minimum and maximum values are strings (in case that the corresponding hyperparameters are strings)
                # bools, or numbers
                try:
                    x_min = min_hyperparam_value-(max_hyperparam_value-min_hyperparam_value)*0.1
                    x_max = max_hyperparam_value+(max_hyperparam_value-min_hyperparam_value)*0.1
                except TypeError:
                    x_min = -0.5
                    x_max = len(set(filtered_combined_runs_df[hyperparam_name]))-0.5

            # In case that x_min=x_max (that happens for min_hyperparam_value==max_hyperparam_value e.g. if there is only one value), 
            # shift them up and down by an arbitrary value (e.g. 0.1)
            if x_min==x_max:
                x_min -= 0.1
                x_max += 0.1

            # Display the global metric optimum as a horizontal line in case it should be shown
            if display_global_optimum:
                ax.hlines(global_opt_valid_metric_value, x_min, x_max, color='k', ls='--', lw=2, label='Global validation metric optimum', zorder=-1)

            # Set x axis limits
            ax.set_xlim([x_min, x_max])
        else:
            # In case the hyperparameter values are non-numeric, rotate the tick labels (as they could be long strings)
            xtick_labels = ax.get_xticklabels()
            ax.set_xticks(list(range(len(xtick_labels))))
            ax.set_xticklabels(xtick_labels, rotation=90)

            # Determine x_min, x_max for the non-numeric values (i.e. their ticks)
            x_min = 0
            x_max = len(xtick_labels)-1
            x_range = x_max-x_min
            x_min -= x_range/10
            x_max += x_range/10

            # Set x axis limits
            ax.set_xlim([x_min, x_max])

            # Display the global metric optimum as a horizontal line in case it should be shown
            if display_global_optimum:
                ax.hlines(global_opt_valid_metric_value, x_min, x_max, color='k', ls='--', lw=2, label='Global validation metric optimum', zorder=-1)

        # Remark: In case the global optimum should be displayed also use it
        #         to determine the minimum/maximum of the y-axis
        if display_global_optimum:
            # If the global optimum should be displayed, differ the metric minimization and maximization cases
            if self.metric_optimization=='min':
                y_min = global_opt_valid_metric_value*0.9
                y_max = fixed_max_valid_metric_value*1.05
            else:
                y_min = fixed_min_valid_metric_value*0.95
                y_max = global_opt_valid_metric_value*1.1
        else:
            y_min = fixed_min_valid_metric_value*0.95
            y_max = fixed_max_valid_metric_value*1.05

        # Set y axis limits
        ax.set_ylim([y_min, y_max])
        ##############################################################################################################################################

        # Display the hyperparameter value for which the validation metric is optimal on for the fixed hyperparameter set
        ax.vlines(fixed_optimal_hyperparam_value, y_min, y_max, color='c', ls=':', lw=1, label='Optimal (for fixed hyperparameters)', zorder=-2)

        # Set the xlabel
        ax.set_xlabel(hyperparam_name, fontsize=axis_fs)

        # In case that the x subplot index is 0 (first column), display the y-axis
        if sp_index_x==0:
            ax.set_ylabel(metric_label, fontsize=axis_fs)

        # For certain columns (=hyperparameters) use a logarithmic x axis
        if hyperparam_name in self.log10_hyperparams:
            # convert y-axis to Logarithmic scale
            ax.set_xscale('log')

        # In case that the current hyperparameter is the first hyperparameter, display a legend
        if hyperparam_index==0:
            ax.legend()

        return y_min, y_max

    def get_save_file_path(self, 
                           quantity_label):
        """
        Return the path to a file in the (save) folder corresponding to the specified quantity label (e.g. 'combined_runs_df'). 

        Args:
            quantity_label (str): Quantity label.

        Return:
            (str): Save file path.
        
        """
        # Check that the quantity label is key of the file name dictionary
        if quantity_label not in self.quantity_label_to_file_names_map:
            err_msg = f"The passed file label '{quantity_label}' is not a key of the map from quantity label to file names."
            raise KeyError(err_msg)

        # Get the file name corresponding to the quantity label
        file_name = self.quantity_label_to_file_names_map[quantity_label]

        # Construct and return the file path
        return str( Path(self.save_folder_path, file_name) )

    def save_quantity(self, 
                      quantity_label, 
                      quantity_obj):
        """
        Save a quantity (e.g. a single attribute or a dictionary of attributes) saved for an instance of this class. 
        
        Args:
            quantity_label (str): Quantity label.
            quantity_obj (dict): Quantity object as dictionary.
                    
        """
        # Get the file path to the file in which the quantity specified by the quantity label (e.g. 'combined_runs_df') is saved in
        save_file_path = self.get_save_file_path(quantity_label)

        # Differ cases for the different quantities
        if quantity_label=='save_attr_dict':
            # Check that the quantity is a dictionaty
            if not isinstance(quantity_obj, dict):
                err_msg = f"The passed quantity label '{quantity_label}' suggests that the quantity object is a dictionary but the passed object is of type '{type(quantity_obj)}'."
                raise TypeError(err_msg)

            # Save the dictionary as pickle file
            with open(save_file_path, 'wb') as file:
                pickle.dump(quantity_obj, file)
            print(f"Saved '{quantity_label}' as .pickle file in {save_file_path}")
        elif quantity_label in ['combined_runs_df', 'raw_runs_df']:
            # Check that the quantity is a dictionaty
            if not isinstance(quantity_obj, pd.DataFrame):
                err_msg = f"The passed quantity label '{quantity_label}' suggests that the quantity object is a pandas.DataFrame but the passed object is of type '{type(quantity_obj)}'."
                raise TypeError(err_msg)

            # Save the pandas.DataFrame as .tsv file
            quantity_obj.to_csv(save_file_path, sep='\t', index=False)
            print(f"Saved '{quantity_label}' as .tsv file in {save_file_path}")
        else:
            err_msg = f"The quantity '{quantity_label}' is not expected."
            raise ValueError(err_msg)

    def load_quantity(self, 
                      quantity_label):
        """ 
        Load a quantity (e.g. a single attribute or a dictionary of attributes) saved for an instance of this class. 

        Args:
            quantity_label (str): Quantity label.
        
        """
        # Get the file path to the file in which the quantity specified by the quantity label (e.g. 'combined_runs_df') is saved in
        save_file_path = self.get_save_file_path(quantity_label)

        # Differ cases for the different quantities
        if quantity_label=='save_attr_dict':
            # The quantity is a dictionary and has been saved as pickle file
            with open(save_file_path, 'rb') as file:
                loaded_dict = pickle.load(file)
            return loaded_dict
        elif quantity_label in ['combined_runs_df', 'raw_runs_df']:
            # The quantity is a pandas.DataFrame and has been saved as .tsv file
            loaded_df = pd.read_csv(save_file_path, sep='\t')

            # Loop over the iterable columns and transform their entries, which are stringified 
            # tuples after loading, to actual tuple objects containing floats.
            for iterable_col_name in self.iterable_col_names:
                if iterable_col_name in loaded_df.columns:
                    loaded_df[iterable_col_name] = loaded_df[iterable_col_name].apply(self.cast_stringified_tuple_to_float_tuple)

            return loaded_df
        else:
            err_msg = f"The quantity '{quantity_label}' is not expected."
            raise ValueError(err_msg)

    def cast_stringified_tuple_to_float_tuple(self, 
                                              str_tuple):
        """
        Cast a stringified tuple (type str) to a tuple containing floats.
        
        Args:
            str_tuple (str): Stringified tuple of the form '(x_1, x_2, x_3, ..., x_N)'
                where x_j are all floats.

        Return:
            (tuple): Tuple of the form (x_1,x_2,x_3,...,x_N().
        
        """
        # Remove the starting as well as ending parentheses of the stringified tuple
        # and remove all white spaces
        str_tuple = str_tuple.lstrip('(')
        str_tuple = str_tuple.rstrip(')')
        str_tuple = str_tuple.strip()
        
        # Split the stringified tuple on commas thus obtaining a list of string items
        list_of_str = str_tuple.split(',')

        # Cast all list element to floats and make the resulting list a tuple  
        return tuple([float(item) for item in list_of_str])
    
    def check_dataframe_equivalence(self, 
                                    df_1, 
                                    df_2, 
                                    rel_eps=1e-4):
        """
        Check if all entries of two pandas.DataFrames df_1 and df_2 are equivalent (for numeric values up to the passed relative precision).
        
        Remark: Relative precision means precision w.r.t. to the magnitude of the values.
                This means that two numeric values v1 and v2 will be considered
                equivalent if " |v1-v2|<=rel_eps*(|v1|+|v2|)/2 " holds

        Args:
            df_1 (pandas.DataFrame): First DataFrame.
            df_2 (pandas.DataFrame): Second DataFrame.
            rel_eps (float): Epsilon-tolerance allowed for entry value deviation.
                (Default: 1e-4)    
        """
        # First check that all column names are the same
        if set(df_1.columns)!=set(df_2.columns):
            err_msg = f"The two pandas.DataFrames do not have the same column names, the first has the column names\n{df_1.colums}\nand the second has the column names\n{df_2.colums}\n"
            raise ValueError(err_msg)

        # Check if number of rows is the same
        if len(df_1)!=len(df_2):
            err_msg = f"The two pandas.DataFrames do not have the same number of rows, the first has {len(df_1)} rows and the second has {len(df_2)} rows."
            raise ValueError(err_msg)

        # Loop over the colums (which are all equivalent) of both DataFrames
        for col_name in df_1.columns:
            # Loop over the rows of the column series objects
            for cell_1, cell_2 in zip(df_1[col_name], df_2[col_name]):
                # # First check that the types of the cell entries are the same
                # if type(cell_1)!=type(cell_2):
                #     err_msg = f"The types of the cells of column '{col_name}' of the two pandas.DataFrames are not the same. For the first it is '{type(cell_1)}' and the second it is '{type(cell_2)}'"
                #     raise ValueError(err_msg)

                # Handle the cases where the cells are both iterables, only one is, or neither is an iterable
                if isinstance(cell_1, (list, tuple, np.ndarray)) and isinstance(cell_2, (list, tuple, np.ndarray)):
                    # Case where the cell entries are iterables:
                    # Cast both entries to numpy arrays
                    cell_1_arr = np.array(cell_1)
                    cell_2_arr = np.array(cell_2)
                    # Construct the absolute difference and the average absolute arrays from the two 
                    # cell entry arrays
                    abs_diff_arr = np.abs(cell_1_arr-cell_2_arr)
                    avg_abs_arr  = (np.abs(cell_1_arr)+np.abs(cell_2_arr))/2

                    # Treat the case where any element of the iterable cell entries differ by more than 
                    # the imposed 'precision'
                    # Remark: When " rel_eps*(|v1|+|v2|)/2<|v1-v2| " values are considered inequivalent
                    if np.any( rel_eps*avg_abs_arr<abs_diff_arr ):
                        err_msg = f"The column {col_name} contains non-identical iterable entries"
                        raise ValueError(err_msg)
                elif isinstance(cell_1, (list, tuple, np.ndarray)) and (not isinstance(cell_2, (list, tuple, np.ndarray))):
                    err_msg = f"One of the cells of column '{col_name}' of the two pandas.DataFrames is an iterable and the other is not. For the first it is '{type(cell_1)}' and the second it is '{type(cell_2)}'"
                    raise TypeError(err_msg)
                elif (not isinstance(cell_1, (list, tuple, np.ndarray))) and isinstance(cell_2, (list, tuple, np.ndarray)):
                    err_msg = f"One of the cells of column '{col_name}' of the two pandas.DataFrames is an iterable and the other is not. For the first it is '{type(cell_1)}' and the second it is '{type(cell_2)}'"
                    raise TypeError(err_msg)
                else:
                    # Neither of the cell entries is an iterable
                    # Case where the cell entries are not iterables:
                    # Try to cast the cell entries to floats (if this is not possible, a ValueError will be thrown)
                    try:
                        cell_1_val = float(cell_1)
                    except ValueError:
                        # Keep the cell entry as it is (no casting)
                        cell_1_val = cell_1
                    try:
                        cell_2_val = float(cell_1)
                    except ValueError:
                        # Keep the cell entry as it is (no casting)
                        cell_2_val = cell_2

                    # Differ the cases where the cell entries are floats (or castable to floats) or not
                    if isinstance(cell_1_val, str) and isinstance(cell_2_val, str):
                        # In case the scalar cell entries are strings, treat the case where the cell
                        # entries are not equivalent
                        if cell_1_val!=cell_2_val:
                            err_msg = f"The column {col_name} contains non-identical string entries"
                            raise ValueError(err_msg)
                    elif isinstance(cell_1_val, str) and (not isinstance(cell_2_val, str)):
                        err_msg = f"One of the cells of column '{col_name}' of the two pandas.DataFrames is a string and the other is not. For the first it is '{type(cell_1)}' and the second it is '{type(cell_2)}'"
                        raise TypeError(err_msg)
                    elif (not isinstance(cell_1_val, str)) and isinstance(cell_2_val, str):
                        err_msg = f"One of the cells of column '{col_name}' of the two pandas.DataFrames is a string and the other is not. For the first it is '{type(cell_1)}' and the second it is '{type(cell_2)}'"
                        raise TypeError(err_msg)
                    else:
                        # Case where the scalar cell entries are floats (or could be cast to floats)
                        # Construct the absolute difference and the average absolute arrays from the two 
                        # cell entry arrays
                        abs_diff = np.abs(cell_1_val-cell_2_val)
                        avg_abs  = (np.abs(cell_1_val)+np.abs(cell_2_val))/2

                        # Treat the case where the cell entries differ by more than the imposed 'precision'
                        # Remark: When " rel_eps*(|v1|+|v2|)/2<|v1-v2| " values are considered inequivalent
                        if rel_eps*avg_abs<abs_diff:
                            err_msg = f"The column {col_name} contains non-identical scalar entries"
                            raise ValueError(err_msg)

# Define a custom exception for the case that a string can not be cast to a date object
class StringNotCastableToDateError(Exception):
    """ Exception raised if a string can not be matched to a date object. """
    def __init__(self, message="String can not be cast to date object."):
        # Initialize the super class with the message
        super().__init__(message)