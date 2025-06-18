# test_set_evaluation.py

# Import public modules
import collections
import copy
import os
import tqdm
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, interpolate

# Import custom modules
from src import model_handler_factory

def get_ensemble_predictions_map(ensemble_predictions_base_path, 
                                 ensemble_base_dir_list_map, 
                                 construct=False,
                                 eps=1e-10):
    """
    Return the ensembles predictions map that is a dictionary mapping different ensembles to
    a their corresponding prediction tables (as pandas.DataFrame) that holds the predictions 
    for each ensemble model for each protein-molecule-pair for both the training and test set.

    Args:
        ensemble_predictions_base_path (str): Base path to the folder in which the different
            prediction tables are stored in.
        ensemble_base_dir_list_map (dict): Dictionary mapping ensembles (=dictionary-key) to
            the path to the base directory in which their trained model dictionaries are
            located in (=dictionary-value).
        construct (bool): Boolean flag specifying if prediction tables should be constructed 
            or loaded from 'ensemble_predictions_base_path' (if already constructed).
            (Default: False)
        eps (float): Small value used to check equality when saving and loading a data frame.
            (Default: 1e-10)

    Return:
        (dict): Dictionary mapping all ensembles to their predictions.

    """
    # Either construct or load the ensemble predictions
    if construct:
        # Construct the ensemble predictions DataFrame
        print('Construct the ensemble prediction DataFrames')
        ensemble_predictions_map = dict()
        for ensemble_name, base_dir_list in ensemble_base_dir_list_map.items():
            ensemble_predictions = construct_model_ensemble_predictions(base_dir_list) # Returns dict with keys 'train' and 'test'
            for set_name in ensemble_predictions:
                ensemble_predictions_map[f'{ensemble_name}_{set_name}'] = ensemble_predictions[set_name]
        # Save the DataFrames
        print('Save all DataFrames:')
        print('-'*100)
        for key, df in ensemble_predictions_map.items():
            print(f'Saving DataFrame {key}')
            file_name = f'{key}.tsv'
            file_path = str(Path(ensemble_predictions_base_path, file_name))
            df.to_csv(file_path, sep='\t', index=False)
            print(f'Saved DataFrame in: {file_path}')
            loaded_df = pd.read_csv(file_path, sep='\t', dtype=df.dtypes.to_dict())
            are_dfs_equal = df.equals(loaded_df)
            if not are_dfs_equal:
                for col in df.columns:
                    # First check for exact equality
                    col_equal = df[col].equals(loaded_df[col])
                    if not col_equal:
                        # If the entries are not exactly equal, check for numeric columns
                        # if they contain entries that are very close
                        if df.dtypes[col] in ['int64', 'float64']:
                            numeric_col_equal = np.all(np.abs(df[col]-loaded_df[col])<=eps)
                            if not numeric_col_equal:
                                err_msg = f"The entries of the saved and loaded column '{col}' are not equal."
                                raise ValueError(err_msg)
                        else:
                            err_msg = f"The entries of the saved and loaded column '{col}' are not equal."
                            raise ValueError(err_msg)
                
            print('-'*100)
        
        print('Saving done')
    else:
        ### Load the pandas.DataFrames with the train and test set entries and predictions
        print('Load all DataFrames:')
        print('-'*100)
        ensemble_predictions_map = dict()
        for file_name in os.listdir(ensemble_predictions_base_path):
            if file_name=='.ipynb_checkpoints':
                continue
            
            # Load the DataFrame    
            file_path = str(Path(ensemble_predictions_base_path, file_name))
            loaded_df = pd.read_csv(file_path, sep='\t')
            print(f'Loaded DataFrame from: {file_path}')
    
            # Assign it to the dict of DataFrames
            # Remark: file_name has the form <key>.tsv
            key = file_name.split('.')[0]
            ensemble_predictions_map[key] = loaded_df
        
        print('Loading done')

    return ensemble_predictions_map

def load_models(models_dict, 
                model_methods=[], 
                generate_train_set_df=False):
    """ 
    Load the models and add them to the models dictionary. 

    Args:
        models_dict (dict): Dictionary mapping models output type (e.g., 'label_1_prob_prediction')
            to where the trained model files (output-folder) is found.
            Example: models_dict = {
                        'label_1_prob_prediction': {
                                'output_folder_path': <path>
                            }
                     }
        model_methods (list): Should we not only predict when loading models but also execute
            additional model methods while doing so?
            (Default: [])
        generate_train_set_df (bool): Boolean flag indicating if the training set should be
            generated or not (e.g., one might only need the test set predictions and thus not
            have to generate the train set).
            (Default: False)

    Return:
        (dict): Dictionary with predictions for different models.
    
    """
    # Loop over the models, define the model handlers and load the models
    for model_name in models_dict:
        print("="*100)
        print(f"Loading Model: {model_name}")
        print("="*100)
        # Define the model handler (and get the updated config dictionary)
        model_handler, config_dict = model_handler_factory.define_model_handler(models_dict[model_name]['output_folder_path'], load_data=True)

        # Load the model
        model_handler.load()

        # Add the model handler and the configuration dictionary to the models dictionary (for the current model)
        models_dict[model_name]['model_handler'] = model_handler
        models_dict[model_name]['config_dict']   = config_dict

        # Generate the train_set_df if requested
        if model_name=='mean_prediction':
            # Predict for the test set
            test_set_dict = model_handler.predict_for_set(set_name='test', data_attributes=['activity_value', 'protein_features', 'protein_id', 'molecule_id'], model_methods=model_methods)

            # Make the test set analysis plots for a 'regression model'
            make_regression_test_set_plots(test_set_dict, config_dict, model_handler)

            # Predict for the train set if requested
            if generate_train_set_df:
                # Predict for the test set
                train_set_dict = model_handler.predict_for_set(set_name='train', data_attributes=['activity_value', 'protein_features', 'protein_id', 'molecule_id'], model_methods=model_methods)
        elif model_name=='label_1_prob_prediction':
            # Predict for the test set
            test_set_dict = model_handler.predict_for_set(set_name='test', data_attributes=['y', 'protein_features', 'protein_id', 'nswcs'], model_methods=model_methods)

            # Make the test set analysis plots for a 'binary classification model'
            #make_binary_classification_test_set_plots(test_set_dict)

            # Generate the train_set_df if requested
            if generate_train_set_df:
                # Predict for the train set
                train_set_dict = model_handler.predict_for_set(set_name='train', data_attributes=['y', 'protein_features', 'protein_id', 'nswcs'], model_methods=model_methods)
        elif model_name=='levels_prob_prediction':
            # Predict for the test set
            test_set_dict = model_handler.predict_for_set(set_name='test', data_attributes=['activity_level_counts', 'protein_features', 'protein_id', 'molecule_id'], model_methods=model_methods)

            # Cast the 2d numpy array dictionary-values to lists of lists
            # Remark: This is a requirement to be able to cast the dictionary to a pandas DataFrame below
            test_set_dict['activity_level_counts'] = [list(item) for item in test_set_dict['activity_level_counts']]
            test_set_dict['prediction']            = [list(item) for item in test_set_dict['prediction']]

            # Make the test set analysis plots for a 'levels probability prediction'
            make_levels_prediction_test_set_plots(test_set_dict, config_dict, model_handler)

            # Generate the train_set_df if requested
            if generate_train_set_df:
                # Predict for the train set
                train_set_dict = model_handler.predict_for_set(set_name='train', data_attributes=['activity_level_counts', 'protein_features', 'protein_id', 'molecule_id'], model_methods=model_methods)

                # Cast the 2d numpy array dictionary-values to lists of lists
                # Remark: This is a requirement to be able to cast the dictionary to a pandas DataFrame below
                train_set_dict['activity_level_counts'] = [list(item) for item in train_set_dict['activity_level_counts']]
                train_set_dict['prediction']            = [list(item) for item in train_set_dict['prediction']]
        else:
            err_msg = f"The expected model names are 'mean_prediction', 'label_1_prob_prediction', or 'levels_prob_prediction'. Got '{model_name}' instead."
            raise ValueError(err_msg)

        # Transform the dictionary-values to the model method dictionary-keys of the test set dictionary of the model methods 
        # from 2d numpy arrays to lists of lists
        for model_method in model_methods:
            test_set_dict[model_method] = [list(item) for item in test_set_dict[model_method]]

            # If train_set_df generation is requested do this also for train
            if generate_train_set_df:
                train_set_dict[model_method] = [list(item) for item in train_set_dict[model_method]]


        # Transform the test set dictionary to a pandas DataFrame and assign it to the models dictionary
        models_dict[model_name]['test_set_df'] = pd.DataFrame(test_set_dict)

        # If train_set_df generation is requested do this also for train
        if generate_train_set_df:
            models_dict[model_name]['train_set_df'] = pd.DataFrame(train_set_dict)
        
        print()

    print("="*100)
    print("="*100)
    print("All models loaded")

    return models_dict

def merge_dataframes(model_init_seed_to_subset_df_map, 
                     set_name):
    """
    Merge DataFrames each representing tables holding predictions for different 
    ensemble models (i.e., models with different initialization seeds) to
    a single DataFrame holding all predictions for all models of an ensemble.

    Args:
        model_init_seed_to_subset_df_map (dict): Dictionary mapping the different
            ensemble model initiailization seeds (i.e., their identifier) to the
            dictionaries that themselve map different subsets (i.e., 'test' or 'train')
            to the model's predictions (as pandas DataFrame) on each subset.
        set_name (str): Name of the subset (i.e., 'test' or 'train') for which we 
            should merge the pandas DataFrames (containing the predictions) for.

    Return:
        (pandas.DataFrame): DataFrame representing a table of the merged prediction entries
            for an ensemble. In addition, this DataFrame will contain an extra column with
            the ensemble average predictions over all models of the ensemble.

    """
    merged_set_df = None
    model_init_seeds = list(model_init_seed_to_subset_df_map.keys())
    model_init_seeds.sort()
    print(f"Merging predictions for all models with init seeds: {model_init_seeds}")
    prediction_cols = list()
    for model_init_seed in tqdm.tqdm(model_init_seeds):
        model_init_seed_set_df = model_init_seed_to_subset_df_map[model_init_seed][set_name].copy(deep=True)
        prediction_col = f'model_{model_init_seed}_prediction'
        prediction_cols.append(prediction_col)
        model_init_seed_set_df.rename(columns={'prediction': prediction_col}, inplace=True)
        if merged_set_df is None:
            merged_set_df = model_init_seed_set_df
            non_prediction_cols = [col for col in merged_set_df.columns if col!=prediction_col]
            num_rows = len(merged_set_df)
        else:
            merged_set_df = pd.merge(merged_set_df, model_init_seed_set_df, how='inner', on=non_prediction_cols)
            if num_rows!=len(merged_set_df):
                err_msg = f"The number of rows has been changed by the merge ({num_rows}->{len(merged_set_df)}) for model_init_seed={model_init_seed}"
                raise ValueError(err_msg)
    
        print(len(merged_set_df))
        print(list(merged_set_df.columns))
        print('-'*100)

    # Determine the ensemble-average over all models (i.e., average the predictions of all models)
    if 0<len(merged_set_df):
        merged_set_df['ensemble_average_prediction'] = merged_set_df[prediction_cols].mean(axis=1)
    else:
        merged_set_df = pd.DataFrame(columns=merged_set_df.columns+['ensemble_average_prediction'])
    return merged_set_df

def construct_model_ensemble_predictions(base_dir_list):
    """
    Load all models of the ensemble and predict for train and test sets for each model.

    Args:
        base_dir_list (list): List holding all the base directory for the trained models
            of each ensemble.

    Return:
        (dict): Dictionary of ensembles (=dictionary-keys) and the corresponding DataFrames 
            representing tables of predictions for each of the models of each of the ensembles
            (dictionary-values).
    
    """
    # Define the to be determined model methods
    to_be_determine_model_methods = [] #['get_x_molecule', 'get_x_combined']
    
    # Loop over the folders in the base directory
    model_init_seed_to_subset_df_map = dict()
    for base_dir in base_dir_list:
        for model_folder in os.listdir(base_dir):
            # Skip .tar.gz files or '.ipynb_checkpoints'
            if model_folder.endswith('.tar.gz') or model_folder=='.ipynb_checkpoints':
                continue
            
            # Load the config.yaml file of the model 
            config_file_path = str(Path(base_dir, model_folder, 'config.yaml'))
            with open(config_file_path, 'r') as file:
                config_dict = yaml.safe_load(file)
    
            # Read out the model init seed
            model_init_seed = config_dict['model']['model_init_seed']

            ### Differ cases for random forest models and non random forest (i.e., GNN) models
            if config_dict['model']['name']=='random_forest':
                ## Case 1: Random forest models
                # Load the data frame containing the predictions for the test set
                test_set_df_file_name = config_dict['model']['pred_test_set_df_save_file_path']
                test_set_df_file_path = str(Path(base_dir, model_folder, test_set_df_file_name))                    
                if not os.path.isfile(test_set_df_file_path):
                    err_msg = f"No file with test set predictions (random forest) found in: {test_set_df_file_path}"
                    raise FileNotFoundError(err_msg)
                    
                test_set_df = pd.read_csv(test_set_df_file_path, sep='\t')
                test_set_df.drop(columns=['features'], inplace=True)
                test_set_df.rename(columns={'label_1_prob_prediction': 'prediction'}, inplace=True)

                # Define an empty data frame for the train set with the same columns as the test set data frame
                train_set_df = pd.DataFrame(columns=test_set_df.columns)
            else:
                ## Case 2: Non random forest (i.e., GNN) models
                # Define the models dictionary for this model type (only a single model, 
                # because there is only one type of model here)
                # Remark: Predictions of our model correspond to the label-1 (i.e., positive label) 
                #         predictions, which has to be specified.
                models_dict = {
                    'label_1_prob_prediction': {
                        'output_folder_path': str(Path(base_dir, model_folder))
                    }
                }
                
                print(f"Loading for model_init_seed={model_init_seed}")
                # Load the models (i.e. only one here) thereby updating the models_dict
                models_dict = load_models(models_dict, 
                                          model_methods=to_be_determine_model_methods, 
                                          generate_train_set_df=True)
                print(f"Loading done!")
    
                # Extract the train and test set DataFrame
                train_set_df = models_dict['label_1_prob_prediction']['train_set_df']
                test_set_df  = models_dict['label_1_prob_prediction']['test_set_df']
    
            # Assign the test_set_df (containing predictions) as
            # dictionary-value to 'scan_var_to_model_dict_map'
            model_init_seed_to_subset_df_map[model_init_seed] = {'train': train_set_df, 'test': test_set_df}
    
            print('-'*100)
            print('-'*100)
            print('-'*100)
            print('-'*100)
            print('-'*100)
    
    # Merge the dataframes
    merged_df_map = dict()
    for set_name in ['train', 'test']:
        merged_df_map[set_name] = merge_dataframes(model_init_seed_to_subset_df_map, set_name)

    return merged_df_map

def get_metrics_df_map_dict(metrics_base_path, 
                            ensemble_predictions_map, 
                            construct=False):
    """
    Return a dictionary mapping ensembles (with dictionary-keys of the form '<ensemble>_test') to pandas 
    DataFrames representing tables holding the values of various metrics (e.g., recall or precision)
    obtained for a range of decision threshold values on the test set.

    Args:
        metrics_base_path (str): Path in which the metric tables are stored in.
        ensemble_predictions_map (dict): Dictionary mapping ensembles to DataFrames representing
            tables holding predictions.
        construct (bool): Boolean flag indicating if the dictionary should be constructed
            or loaded (if it has already been constructed).
            (Default: False)

    Return:
        (dict): Dictionary mapping ensemble/subet pairs
    
    """
    # Either construct or load the metrics DataFrames map?
    if construct:
        metrics_df_map_dict = dict()
        for key in ensemble_predictions_map:
            # Skip the train sets
            if key.endswith('_train'):
                continue
        
            # Generate the metrics DataFrames map
            metrics_df_map_dict[key] = construct_metrics_df_map(ensemble_predictions_map[key])
        
        # Save the DataFrames
        print('Save all DataFrames:')
        print('-'*100)
        # Save the dictionary
        for ensemble_name in metrics_df_map_dict:
            for prediction_name in metrics_df_map_dict[ensemble_name]:
                df = metrics_df_map_dict[ensemble_name][prediction_name]
                print(f'Saving DataFrame {ensemble_name}/{prediction_name}')
                file_name = f'{ensemble_name}%{prediction_name}.tsv'
                file_path = str(Path(metrics_base_path, file_name))
                df.to_csv(file_path, sep='\t', index=False)
                print(f'Saved DataFrame in: {file_path}')
                loaded_df = pd.read_csv(file_path, sep='\t', dtype=df.dtypes.to_dict())
                are_dfs_equal = np.all(np.isclose(df.to_numpy(), loaded_df.to_numpy()))
                print(f'When loading the DataFrame, is it the same as the saved one? {are_dfs_equal}')
                if not are_dfs_equal:
                    for col in df.columns:
                        print(f'df[{col}]==loaded_df[{col}]: {df[col].equals(loaded_df[col])}')
                        
                    raise ValueError("Saved and loaded DataFrames are not Equal!!!")
                print('-'*100)
        
        print('Saving done')
    else:
        ####################################################################################################
        ####################################################################################################
        ### Load the metric DataFrames
        ####################################################################################################
        ####################################################################################################
        print('Load all DataFrames:')
        print('-'*100)
        metrics_df_map_dict = collections.defaultdict(dict)
        for file_name in os.listdir(metrics_base_path):
            if file_name=='.ipynb_checkpoints':
                continue
    
            # Extract the ensemble and prediction name from file_name
            # that has the form file_name=<ensemble_name>%<prediction_name>.tsv
            ensemble_name, prediction_name_and_suffix = file_name.split('%')
            prediction_name = prediction_name_and_suffix.split('.')[0]
            
            # Load the DataFrame    
            file_path = str(Path(metrics_base_path, file_name))
            loaded_df = pd.read_csv(file_path, sep='\t')
            print(f'Loaded DataFrame from: {file_path}')
    
            # Assign it to the dict of dict of DataFrames
            metrics_df_map_dict[ensemble_name][prediction_name] = loaded_df
        
        print('Loading done')

    return metrics_df_map_dict


def construct_metrics_df_map(predictions_df, 
                             decision_threshold_selection='pred_probs'):
    """
    Plot the receiver operator characteristic (ROC) curve, the precission-recall (PR) curve, 
    and the Matthew's Correlation Coefficient (MCC) curve for the passed data 
    and prediction labels.

    Args:
        prediction_df (pandas.DataFrame): DataFrame holding the data labels 
            (='y', i.e., ground truth labels) and predicted probabilities for 
            each of the models of an ensemble.
        decision_threshold_selection (str): How should the decision threshold
            values used to evaluate the metric be selected?
            If 'pred_probs' => Set the decision thresholds to the predicted 
                probabilities, while also adding 0 and 1 as thresholds.
            If 'grid' => Use an equally spaced grid covering [0, 1]

    Return:
        (dict): Dictionary mapping the ensemble's models (=dictionary-keys) to
            their corresponding DataFrames representing tables holding metric
            values for various decision thresholds.
    
    """   
    # Extract the data labels ('y') from the predictions_df
    data_labels = predictions_df['y']
    
    # Determine a map from model (in the ensemble) to its predictions
    prediction_cols = [col for col in predictions_df.columns if col.endswith('prediction')]
    print(prediction_cols)
    predictions_map = {prediction_col: list(predictions_df[prediction_col]) for prediction_col in prediction_cols}
    
    metrics_df_map = dict()
    for pred_label, pred_label_1_probs in predictions_map.items():
        print(pred_label)
        # Generate a list of decision thresholds corresponding to the unique sorted list of predicted label 1 probabilities
        if decision_threshold_selection=='pred_probs':
            decision_thresholds = np.sort(np.unique([0]+pred_label_1_probs+[1]))
        elif decision_threshold_selection=='grid':
            decision_thresholds = np.linspace(0, 1, 500)
        else:
            err_msg = f"The input 'decision_threshold_selection' must be either 'pred_probs' or 'grid', got '{decision_threshold_selection}' instead."
            raise ValueError(err_msg)
    
        # Loop over these decision thresholds, get the rates ('TPR'='recall', 'FPR', and 'PPV'='precision') as 
        # dictionary for each decision threshold, and append the values to a dictionary of lists of these rates
        metrics_dict = collections.defaultdict(list)
        for decision_threshold in tqdm.tqdm(decision_thresholds):
            # Apped the decision threshold
            metrics_dict['decision_threshold'].append(decision_threshold)
            
            # Predict the labels for the current decision threshold
            pred_labels = predict_binary_labels(pred_label_1_probs, decision_threshold=decision_threshold)
    
            # Get the rates as dictionary for the current threshold
            rates_dict = get_binary_rates(data_labels, pred_labels)
    
            # Append the rates to their corresponding lists
            for key, value in rates_dict.items():
                metrics_dict[key].append(value)

            # Determine the Matthew's correlation coefficient for the data and predicted labels and append it to the corresponding list
            MCC = calculate_MCC(data_labels, pred_labels)
            metrics_dict['MCC'].append(MCC)

        metrics_df_map[pred_label] = pd.DataFrame(metrics_dict)

    return metrics_df_map

def predict_binary_labels(predicted_label_1_probs, 
                          decision_threshold=0.5):
    """ 
    Get binary labels based on the predicted probability of positive label 
    (i.e., label 1) and a decision threshold. 
    
    Args:
        predicted_label_1_probs (array-like): Predicted positive label 
            (i.e., label 1) probabilities.
        decision_threshold (float): Decision threshold in [0, 1] to be
            used to determine the predicted labels from the predicted
            probabilities.
            (Default: 0.5)

    Return:
        (numpy.array): Predicted binary labels.
    
    """
    # Initialize the predicted labels as zeros array
    predict_binary_labels = np.zeros_like(predicted_label_1_probs)

    # Assign a label 1 to all values with a label-1 probability above the decision threshold
    predict_binary_labels[np.where(decision_threshold<=predicted_label_1_probs)] = 1

    return predict_binary_labels

# Determine the optimal MCC 
def plot_confusion_matrix_for_decision_threshold(data_labels, 
                                                 predicted_label_1_probs, 
                                                 decision_threshold=0.5, 
                                                 **kwargs):
    """
    Plot the confusion matrix for a certain decision threshold.

    Args:
        data_labels (array-like): Data (i.e., ground truth) labels.
        predicted_label_1_probs (array-like): Predicted positive label
            (i.e., label 1) probabilities.
        decision_threshold (float): Decision threshold in [0, 1] to be
            used to determine the predicted labels from the predicted
            probabilities.
            (Default: 0.5)
        **kwargs (dict): Forwarded to 'plot_confusion_matrix' function.

    """
    # Predict the labels for the current decision threshold based on the predicted label-1 probabilities
    pred_labels = predict_binary_labels(predicted_label_1_probs, decision_threshold=decision_threshold)

    # Get the current axis and make the confusion matrix plot
    # Remark: Plot the decision thresholds in the second row (axs[1, :])
    plot_confusion_matrix(data_labels, pred_labels, **kwargs)


def plot_confusion_matrix(data_labels, 
                          pred_labels, 
                          show_percentages=True, 
                          cmap='OrRd', 
                          title=None, 
                          ax=None, 
                          tick_labels=['N', 'P'],
                          fs_dict={'title': 12, 'axis': 12, 'tick': 10, 'text': 10},
                          show_MCC=False):
    """ 
    Plot the confusion matrix for the passed data and prediction labels. 

    Args:
        data_labels (array-like): Data (i.e., ground truth) labels.
        pred_labels (array-like): Predicted labels.
        show_percentage (bool): Boolean flag indicating if the percentages
            should be shown below the counts within the tiles of the confusion
            matrix.
            (Default: True)
        cmap (str or object): Colormap to be used.
            (Default: 'OrRd')
        title (str or None): Title to be displayed or None if no title should be displayed.
            (Default: None)
        ax (object or None): Axis objects or None (make a figure and use its axis object).
             (Default: None)
        tick_labels (list): List of tick labels (str) to be used.
            (Default: ['N', 'P'])
        fs_dict (dict): Dictionary for different fontsizes.
             (Default: {'title': 12, 'axis': 12, 'tick': 10, 'text': 10})
        show_MCC (bool): Boolean flag indicating if the MCC should be displayed or not.
            (Default: False)

    Return:
        (numpy.array): Return the confusion matrix entries (i.e., tile counts) as numpy array.
        
    """
    # Get the number of different values the labels can take
    size_label_set = max(2, len( set(data_labels) ))

    # Initialize the confusion matrix as quadratic zeros matrix
    confusion_matrix = np.zeros((size_label_set, size_label_set))

    # Loop over the data and prediction labels
    for data_label in range(size_label_set):
        for pred_label in range(size_label_set):
            # Calculate the number of entries that have the current data and prediction max-labels
            # and assign it to the corresponding entry of the confusion matrix
            confusion_matrix[pred_label, data_label] = len( np.where(np.logical_and(data_labels==data_label, pred_labels==pred_label))[0] )

    # In casse the Matthew's correlation coefficient (MCC) should be shown, show it
    if show_MCC:
        # Check that there are only 2 classes (binary) and throw an error if there are more
        if 2<size_label_set:
            err_msg = f"Cannot compute Matthew's correlation coefficient for more than 2 classes (but 'show_MCC' is passed as True)."
            raise ValueError(err_msg)

        # Calculate the MCC and append the information to the title
        MCC = calculate_MCC(data_labels, pred_labels)
        title += f" (MCC={MCC: .2f})"

    # In case that the axis is None, make a figure
    make_fig = False
    if ax is None:
        make_fig = True
        fig      = plt.figure()
        ax       = plt.gca() 

    # Show the title if one is passed (when title is not None)
    if title is not None:
        ax.set_title(title)

    # Scale the confusion matrix so that the largest entry is 1 and plot the confusion matrix
    _confusion_matrix = confusion_matrix/np.max(confusion_matrix)
    ax.imshow(_confusion_matrix, cmap=cmap, vmin=0, vmax=1.7, origin='lower')

    # Loop over the confusion matrix and display the values
    for data_label in range(size_label_set):
        for pred_label in range(size_label_set):
            # Get the current confusion matrix value
            cm_value = int( confusion_matrix[pred_label, data_label] )

            # Create a confusion matrix label that will be shown within each square (=matrix entry) 
            # in the confusion matrix plot
            if show_percentages:
                # Transform the cm_value to a fraction (w.r.t. total number of points in test set/the confusion matrix)
                # and round it to the next integer
                cm_value_rounded = int(np.round(100*cm_value/np.sum(confusion_matrix)))
                
                # Generate a string label from the rounded value, while specially treating the case where the rounded
                # value is zero
                if cm_value_rounded==0:
                    # If the actual value is zero use '0%' as label, otherwise use '<1%' as label
                    if cm_value==0:
                        cm_label = '0%'
                    else:
                        cm_label = '<1%'
                else:
                    cm_label = str(cm_value_rounded) + '%'

                # Add the percentage below the actual number in parentheses
                cm_label = str(int(cm_value)) + '\n(' + cm_label + ')'
            else:
                # Use the numbers as labels
                cm_label = str(int(cm_value))
            
            # Plot the label in the corresponding square
            ax.text(data_label, 
                    pred_label, 
                    cm_label, 
                    color='k', 
                    fontsize=fs_dict['text'], 
                    fontweight='bold', 
                    horizontalalignment='center', 
                    verticalalignment='center')

    ax.set_xlabel('Ground truth', fontsize=fs_dict['axis'])
    ax.set_ylabel('Prediction', fontsize=fs_dict['axis'])
    ax.set_xticks(range(size_label_set))
    ax.set_yticks(range(size_label_set))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(labelsize=fs_dict['tick'])

    # In case a figure has been made for this plot, show it
    if make_fig:
        plt.show()

    return confusion_matrix


def calculate_MCC(data_labels, 
                  pred_labels, 
                  eps=1e-10):
    """ 
    Return the Matthew's Correlation Coefficient (MCC) for the passed data and prediction labels. 

    Args:
        data_labels (array-like): Data (i.e., ground truth) labels.
        pred_labels (array-like): Predicted labels.
        eps (float): Epsilon value used for numerical stability.
            (Default: 1e-10)

    Return:
        (float): MCC value for the data and predicted labels.
    
    """
    # Get the counts (True Positive, True Negative, False Positive, False Negative) for the labels
    counts_dict = get_binary_counts_dict(data_labels, pred_labels)

    # Calculate the numerator and denominator of the MCC, and then the MCC itself
    MCC_numerator     = counts_dict['TN']*counts_dict['TP']
    MCC_numerator    -= counts_dict['FN']*counts_dict['FP']
    MCC_denominator2  =  (counts_dict['TP']+counts_dict['FP'])
    MCC_denominator2 *= (counts_dict['TP']+counts_dict['FN'])
    MCC_denominator2 *= (counts_dict['TN']+counts_dict['FP'])
    MCC_denominator2 *= (counts_dict['TN']+counts_dict['FN'])
    MCC_denominator   = np.sqrt(MCC_denominator2)
    MCC               = MCC_numerator/(MCC_denominator+eps)

    return MCC

def get_binary_rates(data_labels, 
                     pred_labels, 
                     eps=1e-10):
    """ 
    Return different binary rates as dictionary for the passed data and prediction labels. 
    
    Args:
        data_labels (array-like): Data (i.e., ground truth) labels.
        pred_labels (array-like): Predicted labels.
        eps (float): Epsilon value used for numerical stability.
            (Default: 1e-10)

    Return:
        (dict): Dictionary of varius rates obtained for the data and 
            predicted labels.
    
    """
    # Get the counts (True Positive, True Negative, False Positive, False Negative) for the labels
    counts_dict = get_binary_counts_dict(data_labels, pred_labels)

    # Initialize the rates dictionary
    rates_dict = dict()    

    # True positive rate (=recall)
    rates_dict['TPR'] = counts_dict['TP']/(counts_dict['TP']+counts_dict['FN']+eps)        

    # False positive rate
    rates_dict['FPR'] = counts_dict['FP']/(counts_dict['FP']+counts_dict['TN']+eps)

    # Positive predictive value (=precision)
    rates_dict['PPV'] = counts_dict['TP']/(counts_dict['TP']+counts_dict['FP']+eps)

    # Return a dictionary of rates
    return rates_dict

def get_binary_counts_dict(data_labels, 
                           pred_labels):
    """ 
    Return the the counts of True Positive, True Negative, False Positive, and False Negative as dictionary for the passed data and prediction labels. 

    Args:
        data_labels (array-like): Data (i.e., ground truth) labels.
        pred_labels (array-like): Predicted labels.
        
    Return:
        (dict): Dictionary of true positive, true negative, false positive, 
            and false-negative counts obtained for the data and predicted
            labels.
    
    """
    # Initialize the counts dictionary
    counts_dict = dict()

    # Number of true positives (TP)
    counts_dict['TP'] = len( np.where(np.logical_and(data_labels==1, pred_labels==1))[0] )

    # Number of true negatives (TN)
    counts_dict['TN'] = len( np.where(np.logical_and(data_labels==0, pred_labels==0))[0] )

    # Number of false positives (FP)
    counts_dict['FP'] = len( np.where(np.logical_and(data_labels==0, pred_labels==1))[0] )

    # Number of false negatives (FP)
    counts_dict['FN'] = len( np.where(np.logical_and(data_labels==1, pred_labels==0))[0] )

    return counts_dict

def plot_row(focus_metrics_df_map, 
             focus_predictions_df,
             baseline_metrics_df_map,
             baseline_predictions_df,
             axs=None, 
             show_baseline=True,
             lw=1, 
             ls_random='--',
             x_lim=[0, 1],
             y_min_dict={'ROCC': 0, 'PRC': 0, 'MCCC': -1},
             y_ticks_dict={
                'ROCC': [0.00, 0.25, 0.5, 0.75, 1.0], 
                'PRC': [0.00, 0.25, 0.5, 0.75, 1.0], 
                'MCCC': [-1.0, -0.5, 0.0, 0.5, 1.0],
             },
             label_dict={'focus': r'GNN ($\mu\pm\sigma$)', 
                         'baseline': 'Baseline ($\mu\pm\sigma$)', 
                         'random': 'Random'},
             alpha=0.5,
             show_title=True,
             fs_dict={'title': 25, 'axis': 20, 'tick': 15, 'leg': 15},
             color_dict={
                 'focus': r'GNN ($\mu\pm\sigma$)', 
                 'baseline': 'Baseline ($\mu\pm\sigma$)', 
                 'random': 'Random'
             },
             num_interp_steps=1000,
             illustrated_decision_threshold=0.5,
             show_legend=False):
    """
    Plot a 'row' containing the four columns:
    (1) Receiver operator characteristic (ROC) curve
    (2) precission-recall curve (PRC)
    (3) Matthew's Correlation Coefficient curve (MCCC)
    (4) Confusion matrix at a certain threshold
    for prediction done with the models of a single ensemble.

    Args:
         focus_metrics_df_map (dict): Dictionary mapping focus models 
             to their corresponding DataFrames. Each DataFrame corresponds to a table 
             holding the metric values for each of the focus-model ensemble's models 
             evaluated for various decision thresholds.
         focus_predictions_df (pandas.DataFrame): DataFrame respresenting a table holding
             the predictions of the focus-model ensemble's model.
         baseline_metrics_df_map (dict): Dictionary mapping baseline models 
             to their corresponding DataFrames. Each DataFrame corresponds to a table 
             holding the metric values for each of the baseline-model ensemble's models 
             evaluated for various decision thresholds.
         baseline_predictions_df (pandas.DataFrame): DataFrame respresenting a table holding
             the predictions of the baseline-model ensemble's model.
         axs (list or None): List of axis objects or None.
             If None, make a figure and use the figure's list of axis objects as 'axs'.
             (Default: None)
         show_baseline (bool): Should the baseline model be shown in the plot or not?
             (Default: True)
         lw (float): Line width for random model and ensemble model mean.
             (Default=1.0) 
         ls_random (str): Line style of random model.
             (Default: '--')
         x_lim (list): 2-element list with x-limits.
             (Default: [0, 1])
         y_min_dict (dict): Dictionary for the different minimal y-values (i.e., lower
             axis limits) of the different plots.
             (Default: {'ROCC': 0, 'PRC': 0, 'MCCC': -1})
         y_ticks_dict (dict): Dictionary for the different y-ticks to be used in the
             different plots.
             (Default: {'ROCC': [0.00, 0.25, 0.5, 0.75, 1.0], 'PRC': [0.00, 0.25, 0.5, 0.75, 1.0], 'MCCC': [-1.0, -0.5, 0.0, 0.5, 1.0]})
         label_dict (dict): Dictionary for the different labels.
             (Default: {'focus': r'GNN ($\mu\pm\sigma$)', 'baseline': 'Baseline ($\mu\pm\sigma$)', 'random': 'Random'}) 
         alpha (float): Transparency of the 'area' (representing standard deviation
             over the metrics of the ensemble's models).
             (Default: 0.5)
         show_title (bool): Boolean flag indicating if a title should be shown.
             (Default: True)
         fs_dict (dict): Dictionary for different fontsizes.
             (Default: {'title': 25, 'axis': 20, 'tick': 15, 'leg': 15})
         color_dict (dict): Dictionary for different colors.
             (Default: {'focus_mean': 'black', 'focus_area': 'deepskyblue', 'baseline_mean': 'red', 
             'baseline_area': 'orange', 'random': 'silver'})
         num_interp_steps (int): Number of interpolation steps used to interpolate
             the curves.
             (Default: 1000)
         illustrated_decision_threshold (float): Decision threshold shown for the
             confusion matrix.
             (Default: 0.5)
         show_legend (bool): Boolean flag indicating if the legend should be shown.
             (Default: False)

    Return:
        (dict): Dictionary of dictionaries mapping 'AU-ROC' and 'AU-PRC' (dictionary-keys)
            to dictionaries that themselved map the ensemble's model initialization seeds
            (sub-dictionary-keys) to their corresponding AU-values (sub-dictionary-values).
    """        
    # Make the subplots
    make_fig = False
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        make_fig = True

    ###############################################################
    ### (1) Plot the receiver operator characteristic (PR) curve
    ###############################################################
    ax = axs[0]
    if show_title:
        ax.set_title('Receiver operator characteristic (ROC) curve', fontsize=fs_dict['title'])

    # For focus model ensemble
    AU_ROC_dict = plot_roc_curve(which_model='focus',
                                 metrics_df_map=focus_metrics_df_map,
                                 ax=ax,
                                 lw=lw,
                                 alpha=alpha,
                                 color_dict=color_dict,
                                 num_interp_steps=num_interp_steps)

    # For baseline model ensemble
    if show_baseline:
        plot_roc_curve(which_model='baseline',
                       metrics_df_map=baseline_metrics_df_map,
                       ax=ax,
                       lw=lw,
                       alpha=alpha,
                       color_dict=color_dict,
                       num_interp_steps=num_interp_steps)

    # For a random model
    ax.plot([0, 1], [0, 1], color=color_dict['random'], ls=ls_random, lw=lw, zorder=0)

    # Set plot specs
    ax.set_xlabel('False positive rate', fontsize=fs_dict['axis'])
    ax.set_ylabel('True positive rate', fontsize=fs_dict['axis'])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks(y_ticks_dict['ROCC'])
    ax.tick_params(labelsize=fs_dict['tick'])
    ax.set_xlim(x_lim)
    ax.set_ylim([y_min_dict['ROCC'], 1])

    # Make and show the legend if requested
    if show_legend:
        legend_markers = [
                # Make a custom marker of an area with a line in the middle representing mean+-std of the gnn ensemble models
                (
                    mpatches.Rectangle([0, 0], 1, 1, 
                                       alpha=alpha, 
                                       facecolor=color_dict['focus_area'], 
                                       edgecolor=color_dict['focus_area'], 
                                       lw=lw),
                    mlines.Line2D([0], [0], color=color_dict['focus_mean'], lw=lw),
                )
        ]
        if show_baseline:
            legend_markers += [
                # Make a custom marker of an area with a line in the middle representing mean+-std of the baseline ensemble models
                (
                    mpatches.Rectangle([0, 0], 1, 1, 
                                       alpha=alpha, 
                                       facecolor=color_dict['baseline_area'], 
                                       edgecolor=color_dict['baseline_area'], 
                                       lw=lw),
                    mlines.Line2D([0], [0], color=color_dict['baseline_mean'], lw=lw),
                )
            ]
        legend_markers += [
            # Use a simple line for the random model
            mlines.Line2D([0], [0], color=color_dict['random'], ls=ls_random, lw=lw)
        ]
        leg = ax.legend(legend_markers, 
                        [label_dict['focus'], label_dict['baseline'], label_dict['random']], 
                        loc='lower right', 
                        fontsize=fs_dict['leg'])

    ###############################################################
    ### (2) Plot the precision-recall (PR) curve
    ###############################################################
    # Remark: TPR=recall and PPV=precision
    ax = axs[1]
    if show_title:
        ax.set_title('Precision-recall (PR) curve', fontsize=fs_dict['title'])

    # For focus model ensemble
    AU_PRC_dict = plot_pr_curve(which_model='focus',
                                metrics_df_map=focus_metrics_df_map,
                                ax=ax,
                                lw=lw,
                                alpha=alpha,
                                color_dict=color_dict,
                                num_interp_steps=num_interp_steps)

    # For baseline model ensemble
    if show_baseline:
        plot_pr_curve(which_model='baseline',
                      metrics_df_map=baseline_metrics_df_map,
                      ax=ax,
                      lw=lw,
                      alpha=alpha,
                      color_dict=color_dict,
                      num_interp_steps=num_interp_steps)

    # For a random model
    ax.plot([0, 1], [0.5, 0.5], color=color_dict['random'], ls=ls_random, lw=lw, zorder=0)

    # Set plot specs
    ax.set_xlabel('Recall', fontsize=fs_dict['axis'])
    ax.set_ylabel('Precision', fontsize=fs_dict['axis'])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks(y_ticks_dict['PRC'])
    ax.tick_params(labelsize=fs_dict['tick'])
    ax.set_xlim(x_lim)
    ax.set_ylim([y_min_dict['PRC'], 1])

    ###############################################################
    ### Plot the Matthew's Correlation Coefficient (MCC) curve
    ###############################################################
    ax = axs[2]
    if show_title:
        ax.set_title('MCC curve', fontsize=fs_dict['title'])

    # For focus model ensemble
    plot_mcc_curve(which_model='focus',
                   metrics_df_map=focus_metrics_df_map,
                   ax=ax,
                   lw=lw,
                   alpha=alpha,
                   color_dict=color_dict,
                   num_interp_steps=num_interp_steps)

    # For baseline model ensemble
    if show_baseline:
        plot_mcc_curve(which_model='baseline',
                       metrics_df_map=baseline_metrics_df_map,
                       ax=ax,
                       lw=lw,
                       alpha=alpha,
                       color_dict=color_dict,
                       num_interp_steps=num_interp_steps)

    # For a random model
    ax.hlines(0, 0, 1, color=color_dict['random'], ls=ls_random, lw=lw, zorder=0)

    # Set plot specs
    ax.set_xlabel('Decision threshold', fontsize=fs_dict['axis'])
    ax.set_ylabel('MCC', fontsize=fs_dict['axis'])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks(y_ticks_dict['MCCC'])
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks_dict['MCCC']])
    ax.tick_params(labelsize=fs_dict['tick'])
    ax.set_xlim(x_lim)
    ax.set_ylim([y_min_dict['MCCC'], 1])

    # Display MCC estimates at decision threshold for focus model ensemble
    display_mcc_estimate_at_decision_threshold(focus_predictions_df,
                                               illustrated_decision_threshold)

    ###############################################################
    ### Perform ensemble average prediction of the focus model
    ### ensemble for a specified decision threshold
    ###############################################################
    ax = axs[3]
    display_ensemble_prediction_at_decision_threshold(focus_predictions_df,
                                                      illustrated_decision_threshold,
                                                      ax,
                                                      fs_dict)

    if make_fig:
        plt.show()

    return {'AU-ROC': AU_ROC_dict, 'AU-PRC': AU_PRC_dict}


def plot_roc_curve(which_model,
                   metrics_df_map, 
                   ax,
                   lw,
                   alpha,
                   color_dict,
                   num_interp_steps):
    """
    Plot the Receiver operator characteristic (ROC) curve for the specified model ensemble.

    Args:
        which_model (str): For which model (ensemble) to plot the 'curve'. 
            Should be either 'focus' or 'baseline'.
        metrics_df_map (dict): Dictionary mapping models to their corresponding 
            DataFrames. Each DataFrame corresponds to a table holding the metric 
            values for each of the ensemble's models evaluated for various decision 
            thresholds.
        ax (object): Axis object.
        lw (float): Line width for the ensemble mean line.
        alpha (float): Transparency of the 'area' (representing ensemble 
            standard deviation).
        color_dict (dict): Dictionary for different colors.
        num_interp_steps (int): Number of interpolation steps used to 
            interpolate the curves.
        
    Return:
        (dict): Disctionary holding the AU value for each ensemble model.
    """
    if which_model not in ['focus', 'baseline']:
        err_msg = f"The input 'which_model' must be either 'focus' or 'baseline', got '{which_model}' instead."
        raise ValueError(err_msg)
    
    AU_ROC_dict = dict()
    fpr_interp = np.linspace(0, 1, num_interp_steps)
    tpr_interp_list = list()
    for key, metrics_df in metrics_df_map.items():
        inds = np.argsort(metrics_df['FPR'])
        fpr  = [0]+list(metrics_df['FPR'][inds])+[1]
        tpr  = [0]+list(metrics_df['TPR'][inds])+[1]
        
        # Only interpolate if the current predictions are of a model
        # and not of the entire ensemble
        if key!='ensemble_average_prediction':
            f_interp   = interpolate.interp1d(fpr, tpr)
            tpr_interp = f_interp(fpr_interp)
            tpr_interp_list.append(tpr_interp)

        # Calculate the Area Under (AU) the curve
        AUC = np.trapz(tpr, x=fpr)
        AU_ROC_dict[key] = AUC
        
    tpr_interp_matrix = np.vstack(tpr_interp_list) # Shape (#ensemble-models, #decision_thresholds)
    mean_tpr_interp   = np.mean(tpr_interp_matrix, axis=0)
    std_tpr_interp    = np.std(tpr_interp_matrix, axis=0)

    if which_model=='focus':
        print(f"[ROC-Curve]: Maximal standard deviation: {np.max(std_tpr_interp)}")

    # Make plots
    ax.plot(fpr_interp, 
            mean_tpr_interp, 
            color=color_dict[f'{which_model}_mean'], 
            lw=lw,
            zorder=2)
    ax.fill_between(fpr_interp, 
                    y1=mean_tpr_interp-std_tpr_interp, 
                    y2=mean_tpr_interp+std_tpr_interp, 
                    color=color_dict[f'{which_model}_area'], 
                    edgecolor=None, 
                    alpha=alpha,
                    zorder=1)

    return AU_ROC_dict

def plot_pr_curve(which_model,
                  metrics_df_map, 
                  ax, 
                  lw,
                  alpha,
                  color_dict,
                  num_interp_steps):
    """
    Plot the Precision-recall (PR) curve for the specified model ensemble.

    Args:
        which_model (str): For which model (ensemble) to plot the 'curve'. 
            Should be either 'focus' or 'baseline'.
        metrics_df_map (dict): Dictionary mapping models to their corresponding 
            DataFrames. Each DataFrame corresponds to a table holding the metric 
            values for each of the ensemble's models evaluated for various decision 
            thresholds.
        ax (object): Axis object.
        lw (float): Line width for the ensemble mean line.
        alpha (float): Transparency of the 'area' (representing ensemble 
            standard deviation).
        color_dict (dict): Dictionary for different colors.
        num_interp_steps (int): Number of interpolation steps used to 
            interpolate the curves.
        
    Return:
        (dict): Disctionary holding the AU value for each ensemble model.
    """
    if which_model not in ['focus', 'baseline']:
        err_msg = f"The input 'which_model' must be either 'focus' or 'baseline', got '{which_model}' instead."
        raise ValueError(err_msg)
    
    AU_PRC_dict = dict()
    tpr_interp = np.linspace(0, 1, num_interp_steps)
    ppv_interp_list = list()
    for key, metrics_df in metrics_df_map.items():
        inds = np.argsort(metrics_df['TPR'])
        tpr  = [0]+list(metrics_df['TPR'][inds])+[1]
        ppv  = [1]+list(metrics_df['PPV'][inds])+[list(metrics_df['PPV'][inds])[-1]]    
        
        # Only interpolate if the current predictions are of a model
        # and not of the entire ensemble
        if key!='ensemble_average_prediction':
            f_interp   = interpolate.interp1d(tpr, ppv)
            ppv_interp = f_interp(tpr_interp)
            ppv_interp_list.append(ppv_interp)

        # Calculate the Area Under (AU) the curve
        AUC = np.trapz(ppv, x=tpr)
        AU_PRC_dict[key] = AUC
        
    ppv_interp_matrix = np.vstack(ppv_interp_list) # Shape (#models, #decision_thresholds)
    mean_ppv_interp   = np.mean(ppv_interp_matrix, axis=0)
    std_ppv_interp    = np.std(ppv_interp_matrix, axis=0)
    if which_model=='focus':
        print(f"[PR-Curve]:  Maximal standard deviation: {np.max(std_ppv_interp)}")

    # Make plots
    ax.plot(tpr_interp, 
            mean_ppv_interp, 
            color=color_dict[f'{which_model}_mean'], 
            lw=lw,
            zorder=2)
    ax.fill_between(tpr_interp, 
                    y1=mean_ppv_interp-std_ppv_interp,
                    y2=mean_ppv_interp+std_ppv_interp,
                    color=color_dict[f'{which_model}_area'], 
                    edgecolor=None, 
                    alpha=alpha,
                    zorder=1)
    return AU_PRC_dict

def plot_mcc_curve(which_model,
                   metrics_df_map, 
                   ax, 
                   lw,
                   alpha,
                   color_dict,
                   num_interp_steps):
    """
    Plot the Matthew's Correlation Coefficient (MCC) curve for the specified model ensemble.

    Args:
        which_model (str): For which model (ensemble) to plot the 'curve'. 
            Should be either 'focus' or 'baseline'.
        metrics_df_map (dict): Dictionary mapping models to their corresponding 
            DataFrames. Each DataFrame corresponds to a table holding the metric 
            values for each of the ensemble's models evaluated for various decision 
            thresholds.
        ax (object): Axis object.
        lw (float): Line width for the ensemble mean line.
        alpha (float): Transparency of the 'area' (representing ensemble 
            standard deviation).
        color_dict (dict): Dictionary for different colors.
        num_interp_steps (int): Number of interpolation steps used to 
            interpolate the curves.

    """
    if which_model not in ['focus', 'baseline']:
        err_msg = f"The input 'which_model' must be either 'focus' or 'baseline', got '{which_model}' instead."
        raise ValueError(err_msg)
    
    decision_threshold_interp = np.linspace(0, 1, num_interp_steps)
    decision_threshold_interp = decision_threshold_interp[1:-1]
    mcc_interp_list = list()            
    for key, metrics_df in metrics_df_map.items():        
        if key!='ensemble_average_prediction':
            f_interp   = interpolate.interp1d(metrics_df['decision_threshold'], metrics_df['MCC'])
            mcc_interp = f_interp(decision_threshold_interp)
            mcc_interp_list.append(mcc_interp)
        
    mcc_interp_matrix = np.vstack(mcc_interp_list) # Shape (#models, #decision_thresholds)
    mean_mcc_interp   = np.mean(mcc_interp_matrix, axis=0)
    std_mcc_interp    = np.std(mcc_interp_matrix, axis=0)
    if which_model=='focus':
        print(f"[MCC-curve]: Maximal standard deviation: {np.max(std_mcc_interp)}")

    # Make plots
    ax.plot(decision_threshold_interp, 
            mean_mcc_interp, 
            color=color_dict[f'{which_model}_mean'], 
            lw=lw,
            zorder=2)
    ax.fill_between(decision_threshold_interp, 
                    y1=mean_mcc_interp-std_mcc_interp, 
                    y2=mean_mcc_interp+std_mcc_interp, 
                    color=color_dict[f'{which_model}_area'], 
                    edgecolor=None, 
                    alpha=alpha, 
                    zorder=1)

def display_mcc_estimate_at_decision_threshold(predictions_df,
                                               decision_threshold):
    """
    Display the Matthew's Correlation Coefficient (MCC) estimate over 
    the ensemble for a certain decision threshold.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame respresenting a table holding
            the predictions of the ensemble's model.
        decision_threshold (float): Specified decision threshold.

    """
    ### Display the MCC (mean+-standard error) for the models at the specified decision threshold
    # Extract the data labels ('y') from the predictions_df
    data_labels = predictions_df['y']
    
    # Predict the labels for the current decision threshold
    prediction_cols = [col for col in predictions_df.columns if col.endswith('prediction')]
    model_prediction_cols = [prediction_col for prediction_col in prediction_cols if prediction_col!='ensemble_average_prediction']
    MCC_list = list()
    for model_prediction_col in model_prediction_cols:
        pred_label_1_probs = predictions_df[model_prediction_col]
        pred_labels = predict_binary_labels(pred_label_1_probs, 
                                            decision_threshold=decision_threshold)
        MCC = calculate_MCC(data_labels, pred_labels)
        MCC_list.append(MCC)

    # Display results
    print()
    print(f"MCC@decision_threshold={decision_threshold}:")
    print(MCC_list)
    mean = np.mean(MCC_list)
    std  = np.std(MCC_list)
    err  = std/len(MCC_list) # Standard error
    print(f"<MCC>={mean:.4f}+-{err:.4f}")

def display_ensemble_prediction_at_decision_threshold(predictions_df,
                                                      decision_threshold,
                                                      ax,
                                                      fs_dict):
    """
    Display the ensemble prediction at a specified decision threshold and
    plot the corresponding confusion matrix.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame respresenting a table holding
            the predictions of the ensemble's model.
        decision_threshold (float): Specified decision threshold.
        ax (object): Axis object.
        fs_dict (dict): Dictionary for different fontsizes.

    """
    # Extract the data labels ('y') from the predictions_df
    data_labels = predictions_df['y']

    # Prediction labels
    pred_label_1_probs = predictions_df['ensemble_average_prediction']

    # Predict the labels for the current decision threshold
    pred_labels = predict_binary_labels(pred_label_1_probs, 
                                        decision_threshold=decision_threshold)

    # Display the MCC for the ensemble prediction
    MCC_ensemble = calculate_MCC(data_labels, pred_labels)
    print(f" MCC@decision_threshold={decision_threshold}:")
    print(f" MCC (ensemble average): {MCC_ensemble}")
    print()
    print('-'*100)
    
    # Plot the confusion matrix if requested
    plot_confusion_matrix(data_labels, 
                          pred_labels, 
                          ax=ax, 
                          show_percentages=True, 
                          show_MCC=False, 
                          cmap='OrRd',
                          fs_dict=fs_dict,
                          title=None)

def do_hypothesis_test(values_1, 
                       values_2,
                       parametric_hypothesis_test=True, 
                       paired_hypothesis_test=False):
    """
    Do Hypothesis testing on H0:"equality of central tendency".

    Args:
        values_1 (array-like): Values of distribution 1.
        values_2 (array-like): Values of distribution 2.
        parametric_hypothesis_test (bool): Boolean flag indicating
            if a parameteric hypothesis test should be performed.
            (Default: True)
        paired_hypothesis_test (bool): Boolean flag indicating
            if a paired hypothesis test should be performed.
            (Default: False)
        
    """
    if paired_hypothesis_test:
        if parametric_hypothesis_test:
            # Paired t-test
            res = stats.ttest_rel(a=values_1, b=values_2, alternative='two-sided')
        else:
            # Wilcoxon test is non-parametric test for paired samples hypothesis test
            res = stats.wilcoxon(x=values_1, y=values_2, alternative='two-sided', zero_method='wilcox')

    else:
        if parametric_hypothesis_test:
            # 2 independent population t-test
            res = stats.ttest_ind(a=values_1, b=values_2, alternative='two-sided')
        else:
            # Mann-Whitney-U test is non-parametric test for 2 independent populations hypothesis test
            res = stats.mannwhitneyu(x=values_1, y=values_2, alternative='two-sided')

    print(f"p-value (under H0, i.e., equality of central tendency): {res.pvalue*100:.2f}% (={res.pvalue}) [paired-test={paired_hypothesis_test}, parametric-test={parametric_hypothesis_test}]")

def make_figure(ensemble_predictions_map, 
                metrics_df_map_dict, 
                plot_specs, 
                hypothesis_testing_specs):
    """
    Make the main figure displaying the test set analysis.

    Args:
        ensemble_predictions_map (dict): Dictionary mapping ensembles to DataFrames representing
            tables holding predictions.
        metrics_df_map_dict (dict): Dictionary mapping ensembles to pandas DataFrames representing 
            tables holding the values of various metrics (e.g., recall or precision) obtained for 
            a range of decision threshold values.
        plot_specs (dict): Plot specs that are forwarded to 'plot_row' function.
        hypothesis_testing_specs (dict or None): Dictionary holding hypothesis
            testing specs containing the dictionary-keys 'paired_hypothesis_test'
            and 'parametric_hypothesis_test', and corresponding boolean dictionary-values.
            
        
    """
    # Make a figure
    sp_size  = 4 # sub-plot size
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(sp_size*4, sp_size*2.1))
    AU_map   = dict()
    
    # Original model as focus model (top row)
    print('original_test:')
    AU_map['original_test'] = plot_row(focus_metrics_df_map=metrics_df_map_dict['original_test'], 
                                       focus_predictions_df=ensemble_predictions_map['original_test'],
                                       baseline_metrics_df_map=metrics_df_map_dict['baseline_test'], 
                                       baseline_predictions_df=ensemble_predictions_map['baseline_test'],
                                       axs=axs[0, :], 
                                       show_legend=True,
                                       **plot_specs)
    
    # Scrambled model as focus model (bottom row)
    print('scrambled_test:')    
    AU_map['scrambled_test'] = plot_row(focus_metrics_df_map=metrics_df_map_dict['scrambled_test'], 
                                        focus_predictions_df=ensemble_predictions_map['scrambled_test'],
                                        baseline_metrics_df_map=metrics_df_map_dict['baseline_test'], 
                                        baseline_predictions_df=ensemble_predictions_map['baseline_test'],
                                        axs=axs[1, :], 
                                        show_baseline=False, # Baseline model was trained on original dataset
                                        show_legend=True,
                                        **plot_specs)
    
    # Layout
    plt.tight_layout(h_pad=8, w_pad=0.5)
    plt.show()
    
    # Save the figure
    file_name = 'test_set_evaluation.png'
    file_path = str(Path('./saved/figures', file_name))
    fig.savefig(file_path, dpi=300)
    print(f"Saved figure in: {file_path}")
    
    
    ################################################################################################
    ### Make the plot with the resampling model as focus model
    ################################################################################################
    print('resampling scenario:')
    sp_size  = 4 # sub-plot size
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(sp_size*4, sp_size*2.1/2))
    AU_map['resampled_test'] = plot_row(focus_metrics_df_map=metrics_df_map_dict['resampled_test'], 
                                        focus_predictions_df=ensemble_predictions_map['resampled_test'],
                                        baseline_metrics_df_map=metrics_df_map_dict['baseline_test'], 
                                        baseline_predictions_df=ensemble_predictions_map['baseline_test'],
                                        axs=axs,
                                        show_baseline=False, # Baseline model was trained on original dataset
                                        show_legend=True,
                                        **plot_specs)
    plt.tight_layout(h_pad=8, w_pad=0.5)
    plt.show()

    ################################################################################################
    ### Make the plot with the baseline model as focus model
    ### Remark: Here, we pass the base line model in twice:
    ###         (1) As focus model
    ###         (2) As baseline model (default)
    ################################################################################################
    print('*'*100)
    print('*'*100)
    print('*'*100)
    print('baseline model:')
    # Changes for baseline model
    baseline_as_focus_plot_specs = copy.deepcopy(plot_specs)
    baseline_as_focus_plot_specs['label_dict']['focus'] = 'Baseline ($\mu\pm\sigma$)'
    
    sp_size  = 4 # sub-plot size
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(sp_size*4, sp_size*2.1/2))
    AU_map['baseline_test'] = plot_row(focus_metrics_df_map=metrics_df_map_dict['baseline_test'], 
                                       focus_predictions_df=ensemble_predictions_map['baseline_test'],
                                       baseline_metrics_df_map=metrics_df_map_dict['baseline_test'], 
                                       baseline_predictions_df=ensemble_predictions_map['baseline_test'],
                                       axs=axs,
                                       show_legend=True,
                                       **baseline_as_focus_plot_specs)
    plt.tight_layout(h_pad=8, w_pad=0.5)
    plt.show()

    print('*'*100)
    print('*'*100)
    print('*'*100)

    ################################################################################################
    ################################################################################################
    ### Summary statistics and hypothesis testing
    ################################################################################################
    ################################################################################################    
    # AU_map contains AU for all models and ensemble predictions and has the form {ensemble_name: {metric: {<prediction>: AU-value}}}.
    # Construct a dict-of-dicts of the form {metric: {ensemble_name: array([AU-values])} where the values array only
    # contains the AU values of the model predictions (and not of the ensemble prediction)         
    models_AU_map = collections.defaultdict(dict)
    for metric in ['AU-ROC', 'AU-PRC']:
        for ensemble_name in AU_map:
            # Get all model names (i.e., the ensemble average prediction) and sort them
            # Remark: Sorting the model names will ensure that the order of the AU-values
            #         (extracted below) is the same for any metric/ensemble-name combination
            #         that is important for hypothesis testing (below)
            model_names = [key for key in AU_map[ensemble_name][metric] if key!='ensemble_average_prediction']
            model_names.sort()
    
            # Get the AU values of all models
            AU_values = [AU_map[ensemble_name][metric][model_name] for model_name in model_names]
            models_AU_map[metric][ensemble_name] = np.array(AU_values)
    
    for metric in models_AU_map:
        print(f"Metric: {metric}\n")
        AU_values_dict = dict()
    
        # Show mean and standard errors
        for ensemble_name in models_AU_map[metric]:
            print(f" Ensemble: {ensemble_name}")
            AU_values = models_AU_map[metric][ensemble_name]
            print(f" {AU_values}")
            mean = np.mean(AU_values)
            std  = np.std(AU_values)
            err  = std/len(AU_values) # Standard error
            print(f" <AU>={mean:.4f}+-{err:.4f}")
            AU_values_dict[ensemble_name] = AU_values
            print(f" AU (ensemble average): {AU_map[ensemble_name][metric]['ensemble_average_prediction']}")
            print()

        # For AU-ROC do hypothesis tests
        if metric=='AU-ROC':
            # Test optimal model trained on the original training set vs. 
            # the ones trained on the scrambled and resampled training set
            print("Original vs. scrambled:")
            do_hypothesis_test(AU_values_dict['original_test'], 
                               AU_values_dict['scrambled_test'],
                               parametric_hypothesis_test=hypothesis_testing_specs['parametric'], 
                               paired_hypothesis_test=hypothesis_testing_specs['paired'])
            print()
            print("Original vs. resampled:")
            do_hypothesis_test(AU_values_dict['original_test'], 
                               AU_values_dict['resampled_test'],
                               parametric_hypothesis_test=hypothesis_testing_specs['parametric'], 
                               paired_hypothesis_test=hypothesis_testing_specs['paired'])
            print()
            print("Original vs. baseline:")
            do_hypothesis_test(AU_values_dict['original_test'], 
                               AU_values_dict['baseline_test'],
                               parametric_hypothesis_test=hypothesis_testing_specs['parametric'], 
                               paired_hypothesis_test=hypothesis_testing_specs['paired'])
        
            print()
            print('-'*100)
            print()
