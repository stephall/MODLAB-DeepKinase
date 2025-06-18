# model_handling_rf.py

# Import public modules
import time
import joblib
import pickle
import tqdm
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Import custom modules
from . import utils
from . import random_handling

class RFModelHandler(object):    
    def __init__(self, 
                 data_handler, 
                 config_dict, 
                 logger=None):
        """
        Initialize model handler that is used to govern both the model, but also the data
        it is trained and evaluated on.
        
        Args:
            data_handler (src.data_handling.DataHandler): Data handler object.
            config_dict (dict): Dictionary containing the configurations.
            logger (logging.logger or None): Logger object or None.
                (Default: None)

        """
        # Assign inputs to class attributes
        self.data_handler                = data_handler
        self.logger                      = logger
        self.config_dict                 = config_dict
        self.raw_data_base_dir           = self.config_dict['raw_base_data_dir']
        self.rf_hyper_params_scan_values = self.config_dict['rf_hyper_params_scan_values']
        self.display_info(self.config_dict)

        ### Load the nswcs to feature maps
        self.which_feature_types = list()
        self._nswcs_to_feature_maps_dict = dict()

        ### Fingerprint (as molecular features)
        self.display_info(f"Molecular fingerprint: {self.config_dict['mol_fp_label']}")
        file_name = f"nswcs_to_{self.config_dict['mol_fp_label']}_map.pickle"
        file_path = Path(self.raw_data_base_dir, self.config_dict['fp_mappings_rel_dir_path'], file_name)
        if not os.path.isfile(file_path):
            err_msg = f"No mappings file for mol_fp_label={self.config_dict['mol_fp_label']} found in {file_path}"
            raise ValueError(err_msg)
        
        self._nswcs_to_feature_maps_dict['mol_fp'] = self._load_pickle_file(file_path)
        self.which_feature_types.append('mol_fp')
        
        ### Get the train and test set and add the features to their smiles strings
        # Get a dictionary containing the different sets
        subset_df_map = data_handler.data_preprocessor.set_to_processed_subset_df_map

        # Check that the validation set is empty (i.e. train contains whole train-validation set)
        if 0<len(subset_df_map['valid']):
            err_msg = f"We expect an empty validation set for random forest models, however the validation set is not empty!"
            raise ValueError(err_msg)
        
        # Add the features to the train and test set
        self.train_set_df = self._add_features_to_data_df(subset_df_map['train'], which_features=self.which_feature_types)
        self.test_set_df  = self._add_features_to_data_df(subset_df_map['test'], which_features=self.which_feature_types)
        self.display_info(f"Number of features: {len(self.train_set_df['features'][0])}")

        ### Construct the list of hyper-parameter combinations that should be scanned for during tuning
        self.hyper_params_list = self._construct_hyper_params_list()

        ### Initialize empty proteins model dictionary
        self.protein_model_dict = dict()

    def display_info(self, 
                     info_msg):
        """
        Display information either logging it or by printing it if no logger is defined (self.logger is None).

        Args:
            info_msg (str): Information to be displayed.
        
        """
        if self.logger is not None:
            self.logger.info(info_msg)
        else:
            print(info_msg)

    def train(self, 
              num_epochs=None):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs that is passed (for reasons of 
                coherency with the neural network model handler) but not 
                needed for a random forest model.
        """
        # As we want to use single-task base line models, create a dictionary
        # that contains the proteins as dictionary-keys and a dictionary with
        # the X and y values for both the train and test set
        protein_ids = list(set(self.train_set_df['protein_id']))
        utils.sort_chembl_id_list(protein_ids)
        pred_protein_test_df_list = list()
        start_time_outer = time.time()
        for protein_index, protein_id in enumerate(protein_ids):            
            self.display_info('-'*100)
            start_time = time.time()

            # Get the subsets of the data belonging to the protein
            protein_train_set_df = self.train_set_df[self.train_set_df['protein_id']==protein_id]
            protein_test_set_df  = self.test_set_df[self.test_set_df['protein_id']==protein_id]

            # Get the (binary) y-values
            y_train = np.array(protein_train_set_df['y'])

            self.display_info(f"Protein ID: {protein_id} ({protein_index+1}/{len(protein_ids)})")
            self.display_info(f"(#train-validation-points: {len(y_train)})")

            ################################################################################################
            ### Step 1: Tune hyper parameters using cross-validation
            ################################################################################################
            # Get the X-values (i.e. fingerprints)
            X_train = np.vstack(protein_train_set_df['features']).astype(np.float64)
            X_test  = np.vstack(protein_test_set_df['features']).astype(np.float64)

            # Train for all hyper parameters
            tuning_dict_list = self._train_for_hyper_params(X_train, y_train, self.hyper_params_list, model_init_seed=self.config_dict['model_init_seed'], **self.config_dict['rf_cv_params'])

            ################################################################################################
            ### Step 2: Train the optimal model for the current protein
            ################################################################################################
            # Find the best hyper parameter value combination (i.e. the optimal tuning dictionary)
            optimal_tuning_dict = min(tuning_dict_list, key=lambda x: x['metric_value'])

            self.display_info(f"Optimal hyper parameters: {optimal_tuning_dict['hyper_params']}")
            self.display_info(f"Optimal metric value: {optimal_tuning_dict['metric_value']}")

            # Train model for the hyper parameters in the optimal tuning dictionary
            # and assign it as dictionary-value to the protein ID as dictionary-key
            self.protein_model_dict[protein_id] = self._train_optimal_model(X_train, y_train, optimal_tuning_dict['hyper_params'], model_init_seed=self.config_dict['model_init_seed'])

            # Predict the class-probabilities with the model on the test data
            prob_pred = self.protein_model_dict[protein_id].predict_proba(X_test)  # Shape (batch_size, 2) 

            # Extract the label-1 probabilities and add them as new column to the test DataFrame
            pred_protein_test_set_df = protein_test_set_df.copy(deep=True)
            pred_protein_test_set_df['label_1_prob_prediction'] = prob_pred[:, 1]
            pred_protein_test_df_list.append(pred_protein_test_set_df)

            self.display_info(f"Duration: {time.time()-start_time: .2f}s")
            self.display_info('-'*100)
            self.display_info('')

        self.display_info(f"Done training models for all proteins. Duration: {(time.time()-start_time_outer)/60:.2f} mins.")

        # Generate one large test DataFrame
        pred_test_df = pd.concat(pred_protein_test_df_list)

        ## Save quantities
        # Save the prediction DataFrame
        file_path = self.config_dict['pred_test_set_df_save_file_path']
        pred_test_df.to_csv(file_path, sep='\t', index=False)
        self.display_info(f"Saved DataFrame holding predictions in the file {file_path}")

        # Save the optimal protein models
        os.makedirs(self.config_dict['protein_models_save_dir'])
        for protein_id, model in self.protein_model_dict.items():
            file_name = f"{protein_id}_model.joblib"
            file_path = Path(self.config_dict['protein_models_save_dir'], file_name)
            joblib.dump(model, file_path, compress=self.config_dict['model_saving_compression_level'])

        self.display_info(f"Saved the protein models individually in the folder {self.config_dict['protein_models_save_dir']}")
        

    def plot_learning_curve(self, 
                            epoch=None, 
                            **kwargs):
        """
        Plot the learning curve that is not shown for random forest models.
        
        Args:
            epoch (int, str or None): The epoch the learning curve should be plotted 
                for that is passed (for reasons of coherency with the neural network 
                model handler) but not needed for a random forest model.
            **kwargs: Passed but not needed for reasons of coherency with the neural 
                network model handler.
                

        """
        self.display_info("Learning curves are not plotted for random forest models.")

    def _load_pickle_file(self, 
                          file_path):
        """
        Load a pickle file based on the passed file path.

        Args:
            file_path (str): Path to the pickle file.

        Return:
            (object): The loaded file content.
        """
        # Open the file in binary mode and load content 
        with open(file_path, 'rb') as file:  
            # Call load method to deserialze 
            loaded_file = pickle.load(file) 

        return loaded_file

    def _add_features_to_data_df(self, 
                                 data_df, 
                                 which_features):
        """
        Add features to the data dictionary.

        Args:
            data_df (pandas.DataFrame): Data frame to which features
                should be added to.
            which_features (list): List of features (as strings).
                Example: which_features = ['mol_fp']

        Return:
            (pandas.DataFrame): Copy of input data frame to which
                the (molecular) features have been added to.
        
        """
        data_dict = dict()

        # Add key-value pairs based on columns of the data_df
        data_dict['protein_id'] = list(data_df['protein_id'])
        data_dict['nswcs']      = list(data_df['non_stereochemical_washed_canonical_smiles'])
        data_dict['y']          = [1 if item==True else 0 for item in data_df['on_chembl']]

        # Add a key-value pair for the feature column
        data_dict['features'] = [self._get_features(nswcs, which_features) for nswcs in data_dict['nswcs']]

        return pd.DataFrame(data_dict)
    
    def _get_features(self, 
                      nswcs, 
                      which_features):
        """
        Return the molecular features of a molecule's non-stereochemical 
        washed canonical SMILES (nswcs) string.
        
        Args:
            nswcs (str): Nswcs of a molecule.
            which_features (list): List of features (as strings).
                Example: which_features = ['mol_fp']

        Return:
            (tuple): Molecular features as tuple.
        
        """
        features = list()
        for feature_name in which_features:
            # The features are tuples, cast them to list and add them
            features += list(self._nswcs_to_feature_maps_dict[feature_name][nswcs])

        return tuple(features)

    def _construct_hyper_params_list(self):
        """
        Construct the list of hyper parameter combinations.

        Return:
            (list): List of dictionaries where each dictionary represents
                a different combination of hyper parameters.
        """
        # Generate the hyper parameter list
        hyper_params_list = list()
        for criterion_value in self.rf_hyper_params_scan_values['criterion_values']:
            for n_estimators_value in self.rf_hyper_params_scan_values['n_estimators_values']:
                for max_depth_value in self.rf_hyper_params_scan_values['max_depth_values']:
                    for max_features_value in self.rf_hyper_params_scan_values['max_features_values']:
                        for max_samples_value in self.rf_hyper_params_scan_values['max_samples_values']:
                            for min_samples_split_value in self.rf_hyper_params_scan_values['min_samples_split_values']:
                                for min_samples_leaf_value in self.rf_hyper_params_scan_values['min_samples_leaf_values']:
                                    hyper_params_list.append({
                                        'criterion': criterion_value, 
                                        'n_estimators': n_estimators_value,
                                        'max_depth': max_depth_value,
                                        'max_features': max_features_value,
                                        'max_samples': max_samples_value,
                                        'min_samples_split': min_samples_split_value,
                                        'min_samples_leaf': min_samples_leaf_value,
                                    })

        return hyper_params_list

    def _train_for_hyper_params(self, 
                                X_train, 
                                y_train, 
                                hyper_params_list, 
                                model_init_seed=43, 
                                split_seed=42, 
                                num_kfolds=5, 
                                display_progress=True):
        """
        Train the model for all hyper parameter combinations.

        Args:
            X_train (numpy.array): X values of the train set as numpy array of 
                shape (#data-points, #features).
            y_train (numpy.array): y values of the train set as numpy array of
                shape (#data-points,).
            hyper_params_list (list): List of dictionaries that each hold a set
                of model hyper parameters.
            model_init_seed (int): Model initialization seed.
                (Default: 43)
            split_seed (int): Seed for the stratified splitting for in K-fold 
                cross validation..
                (Default: 42)
            num_kfolds (int): Number of folds in K-fold cross validation.
                (Default: 5)
            display_progress (bool): Boolean flag if the progress should be 
                displayed or not.
                (Default: True)

        Return:
            (list): List of dictionaries that hold the information and results
                for each of the hyper parameter combinations (i.e., tuning points).

        """
        # Loop over hyper-parameters
        tuning_dict_list = list()
        if display_progress==True:
            iterator = tqdm.tqdm(hyper_params_list)
        else:
            iterator = hyper_params_list

        for hyper_params in iterator:
            # Define the model factory
            model_factory = lambda : RandomForestClassifier(**hyper_params, random_state=model_init_seed)

            # Loop over the proteins
            metric_value_sum_list = list()
            num_valid_points_list = list()

            # Define stratified K-fold (skf) splitting object
            skf_split_obj = StratifiedKFold(n_splits=num_kfolds, random_state=split_seed, shuffle=True)

            # Train on the folds and return the sum of the metric over all points of all folds
            # and also the number of points validated on over all folds
            metric_value_sum_protein, num_valid_points_protein = self._train_on_folds(skf_split_obj, 
                                                                                      X_train, 
                                                                                      y_train, 
                                                                                      model_factory)
            metric_value_sum_list.append(metric_value_sum_protein)
            num_valid_points_list.append(num_valid_points_protein)

            # Determine the metric value for the current hyper parameter as the average over all validation
            # points of all proteins and all of their folds
            metric_value_hp = np.sum(metric_value_sum_list)/np.sum(num_valid_points_list)

            tuning_dict_list.append({'hyper_params': hyper_params, 'metric_value': metric_value_hp})

        return tuning_dict_list
    
    def _train_on_folds(self, 
                        skf_split_obj, 
                        X_train_valid, 
                        y_train_valid, 
                        model_factory, 
                        eps=1e-10):
        """
        Train the model on all folds.

        Args:
            skf_split_obj (object): StratifiedKFold object of scikit-learn module.
            X_train_valid (numpy.array): X values of the train set (that will be 
                considered and split into a train+validation set here) as numpy 
                array of shape (#data-points, #features).
            y_train_valid (numpy.array): y values of the train set (that will be 
                considered and split into a train+validation set here) as numpy 
                array of shape (#data-points,).
            model_factory (function): Model factory function that returns a model when called.
            eps (float): Small value used for numerical stability in logarithm conputations.
                (Default: 1e-10)

        Return:
            (float, int): Sum over all the metric values of all folds and the total number 
                of validation points 

        """
        # Loop over the K-fold splits, train the model on the train-subset of the 
        # split and evaluate it on the validation-subset of the split
        metric_values_list    = list()
        num_valid_points_list = list()
        for k, (train_indices, valid_indices) in enumerate(skf_split_obj.split(X_train_valid, y_train_valid)):
            # Get the train- and validation-subsets of the current split
            X_train = X_train_valid[train_indices]
            y_train = y_train_valid[train_indices]
            X_valid = X_train_valid[valid_indices]
            y_valid = y_train_valid[valid_indices]

            # Create model object
            model = model_factory()

            # Training the model on the train-subset
            model.fit(X_train, y_train) 

            # Predict on the validation subset 
            y_prob_pred = model.predict_proba(X_valid)
            binary_cross_entropy = -(1-y_valid)*np.log(y_prob_pred[:, 0]+eps) - y_valid*np.log(y_prob_pred[:, 1]+eps)

            # Determine the metric for the current fold
            metric_value_fold = np.sum(binary_cross_entropy)
            metric_values_list.append(metric_value_fold)

            # Determine the number of valuation points
            num_valid_points = X_valid.shape[0]
            num_valid_points_list.append(num_valid_points)

        # Sum the metric values of the folds to obtain the metric of the current
        # hyper-parameter combination and return it together with the total number
        # of validation points.
        return np.sum(metric_values_list), int(np.sum(num_valid_points))

    def _train_optimal_model(self, 
                             X_train, 
                             y_train, 
                             optimal_hyper_params, 
                             model_init_seed=43):
        """
        Train the model for the optimal hyper parameters.

        Args:
            X_train (numpy.array): X values of the train set as numpy array of 
                shape (#data-points, #features).
            y_train (numpy.array): y values of the train set as numpy array of
                shape (#data-points,).
            optimal_hyper_params (dict): Optimal hyper parameters.
            model_init_seed (int): Model initialization seed.
                (Default: 43)

        Return:
            (object): The trained optimal model.

        """
        # Create model object
        model = RandomForestClassifier(**optimal_hyper_params, random_state=model_init_seed)

        # Training the model on the train-subset
        model.fit(X_train, y_train) 

        return model

    def load_protein_models(self, 
                            protein_models_save_dir=None):
        """
        Load the protein specific models.

        Args:
            protein_models_save_dir (str): Path to the directory in which
                the protein specific models are saved in.
        """
        # Use the protein models save directory in the configurations as the default directory
        if protein_models_save_dir is None:
            protein_models_save_dir = self.config_dict['protein_models_save_dir']

        # Throw an error if 'self.protein_model_dict' is not empty
        if 0<len(self.protein_model_dict):
            err_msg = f"The handler already holds protein models and thus cannot load new ones (as this would overwrite the current ones)."
            raise ValueError(err_msg)

        # Load the protein models from all .joblib files
        for file_name in os.listdir(protein_models_save_dir):
            if file_name.endswith('.joblib'):
                file_path = Path(protein_models_save_dir, file_name)
                # Extract the protein ID from the file name that should have the form '<protein_id>_model.joblib'
                protein_id = file_name.split('_')[0]
                self.protein_model_dict[protein_id] = joblib.load(file_path)