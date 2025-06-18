# data_preprocessing.py

# Import public modules
import copy
import collections
import os
import pickle
import re
import torch
import torch_geometric
import tqdm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import custom modules
from . import utils
from . import smiles_to_graph_mapping
from . import protein_sequence_tokenizer

class DataPreprocessor(object):
    def __init__(self, 
                 config_dict, 
                 logger=None, 
                 make_logs=True, 
                 make_figs=True):
        """
        Args:
            config_dict (dict): Config dictionary.
            logger (logging.logger or None): Logger object or None.
                (Default: None)
            make_logs (bool): Boolean flag indicating if logs should 
                be made and saved.
                (Default: True)
            make_figs (bool): Boolean flag indicating if figures 
                should be made and saved.
                (Default: True)
        """
        # Assign inputs to class attributes of the same name
        self.config_dict = config_dict
        self.logger      = logger
        self.make_logs   = make_logs
        self.make_figs   = make_figs

        # Throw and error in case that the relation representation type is not 'binary_labels'
        # Remark: This is necessary because this data preprocessor can only represent the relations as binary labels.
        if self.config_dict['relation_representation_params']['type']!='binary_labels':
            err_msg = f"The relation representation type specified in the configurations ('data_preprocessing.relation_representation_params.type') must be 'binary_labels', but got '{self.config_dict['relation_representation_params']['type']}' instead."
            raise ValueError(err_msg)

        # Define a Smiles to Graph Mapper object
        self.smiles_to_graph_mapper = smiles_to_graph_mapping.SmilesToGraphMapper()

        # Define file paths as class attributes
        self.raw_data_base_dir                      = self.config_dict['raw_data_base_dir']
        self.smiles_to_graph_map_file_path          = Path(self.raw_data_base_dir, self.config_dict['file_names']['smiles_to_graph_map_file'])
        self.measurements_file_path                 = Path(self.raw_data_base_dir, self.config_dict['file_names']['measurements_file'])
        self.molecules_from_qmugs_summary_file_path = Path(self.raw_data_base_dir, self.config_dict['file_names']['molecules_from_qmugs_summary_file'])
        self.nswc_smiles_kinases_chembl_file_path   = Path(self.raw_data_base_dir, self.config_dict['file_names']['nswc_smiles_kinases_chembl_file'])
        self.nswc_smiles_holdout_file_path          = Path(self.raw_data_base_dir, self.config_dict['file_names']['nswc_smiles_holdout_file'])
        self.figures_dir_path                       = self.config_dict['figures_dir_path']

        # In case this directory doesn't already exist, make it
        if not os.path.isdir(self.figures_dir_path):
            os.makedirs(self.figures_dir_path)

        # Assign some selected configurations to class attributes
        self.K = config_dict['K']
        self.k = self.parse_k(config_dict['k'])

        # The data folds are the cross validation (cv) folds (specified by their total number K) and by one additional fold corresponding to the test set 
        # (that should have approx. same size as the folds).
        # The validation set will correspond to one of the cross-validation folds (specified by k) and the train set to all the other cross-validation folds.
        # Thus, we have the data folds [<cv_1>, <cv_2>, <cv_3>, ..., <cv_K>, <test>] of total number K+1.
        self.num_data_folds = self.K+1

        # Ensure that the number of data folds (crossvalidation folds plus one fold for the test set) is bigger than 2
        if self.num_data_folds<2:
            err_msg = f"The input 'num_data_folds' (number of cross-validation folds plus one test fold) must be an integer bigger or equal to 2, got value '{self.num_data_folds}' instead."
            raise ValueError(err_msg)

        # Initialize attributes to None that will be defined later  
        self.measurements_df                = None
        self.filtered_measurements_df       = None
        self.non_measured_nswcs_list        = None
        self.smiles_to_graph_map            = None
        self.smiles_mapped_to_graphs_saved  = None
        self.preprocessed_df                = None
        self.protein_representation_params  = None
        self.set_to_data_fold_indices_map   = None
        self.set_to_processed_subset_df_map = None
        self.set_to_torch_data_list_map     = None

        # Load the measurements (table)
        self._load_measurements_df()

    def parse_k(self, 
                k):
        """
        Parse (during initialization) the the cross-validation (CV) fold index 'k'. 
        
        Args:
            k (int, str, or None): Initialization input CV-fold index.
                Cases:
                    (1) If None or 'None' => Return None
                    (2) If int => Check that it is in [0, self.K-1] (i.e., a valid CV-fold index).

        Return:
            (int or None) Parsed cross-validation fold index.
        
        """
        # Differ cases, where k is either None (or 'None') or where k is a number (represented as int, float, or string)
        if (k is None) or (k=='None'):
            return None
        elif isinstance(k, (int, float, str)):
            # Cast k to an integer (or at least try it)
            k = int(k)
            
            # Check that k is in the allowed range [0, K-1]
            if not (0<=k and k<self.K):
                err_msg = f"The (zero-based) cross-validation fold index k must be an integer in [0, {self.K-1}], got value {k} instead."
                raise ValueError(err_msg)

            return k
        else:
            err_msg = f"Configuration 'k' must be either None (or 'None') or an integer number (represented as int, float, or string) in [0, {self.K-1}]."
            raise TypeError(err_msg)

    def display_info(self, 
                     info_msg):
        """ 
        Display information either logging it or by printing it if no logger is defined (self.logger is None). 

        Args:
            info_msg (str): Information to be displayed.
        
        """
        # Remark: If 'self.make_logs' is False, don't do anything
        if self.make_logs:
            if self.logger is None:
                # If the logger is NOT defined, print the information message
                print(info_msg)
            else:
                # If the logger is defined, log the information message
                self.logger.info(info_msg)

    def display_summary(self, 
                        filtered_df):
        """ 
        Display some summary information about the passed filtered DataFrame. 

        Args:
            filtered_df (pandas.DataFrame): Pandas DataFrame holding filtered entries.
        
        """
        protein_df = filtered_df[filtered_df['protein_id']=='CHEMBL3267']
        self.display_info(f"#entries for protein PK3CG_HUMAN: {len(protein_df)}")

        # Log some general information about the filtered dataset
        self.display_info(f"Number of proteins:               {len( set(filtered_df['protein_id']) )}")
        self.display_info(f"Number of molecules:              {len( set(filtered_df['non_stereochemical_washed_canonical_smiles']) )}")
        self.display_info(f"Number of molecule-protein pairs: {len( set(zip(filtered_df['non_stereochemical_washed_canonical_smiles'], filtered_df['protein_id'])) )}")
        self.display_info(f"Number of activity values:        {len(filtered_df)}\n")

    def load_smiles_to_graph_map(self):
        """ Load the map (dictionary) from (non-stereochemical washed canonical) SMILES strings to graphs. """
        # If the (pickle) file that stores this map exists, load and assign it to the corresponding class attribute,
        if os.path.isfile(self.smiles_to_graph_map_file_path):
            self.display_info(f"Loading the 'smiles_to_graph_map' from the file: {self.smiles_to_graph_map_file_path} ...")
            with open(self.smiles_to_graph_map_file_path, 'rb') as file:
                self.smiles_to_graph_map = pickle.load(file)
            
            self.display_info(f"Loading done.\n")
        else:
            # Otherwise, assign an empty dictionary to the corresponding class attribute,
            self.smiles_to_graph_map = {}

            self.display_info(f"The file {self.smiles_to_graph_map_file_path} does not exist, creating empty 'smiles_to_graph_map'.\n")

        # Make a (deep) copy of the currently saved dictionary-keys (i.e. smiles) and assign them to the corresponding class attribute
        self.smiles_mapped_to_graphs_saved = copy.deepcopy(list(self.smiles_to_graph_map.keys()))

    def save_smiles_to_graph_map(self):
        """ Save the map (dictionary) from (non-stereochemical washed canonical) SMILES strings to graphs. """
        # Only save the current internal map (in a pickle file) if it is different from the original one
        # Remark: The mapping should be injective (i.e. one-to-one) thus check the dictionary-keys (i.e. nswcs) for inequality
        if set(self.smiles_to_graph_map.keys())!=set(self.smiles_mapped_to_graphs_saved):
            self.display_info(f"Saving the 'smiles_to_graph_map' to the file: {self.smiles_to_graph_map_file_path} ...")
            with open(self.smiles_to_graph_map_file_path, 'wb') as file:
                pickle.dump(self.smiles_to_graph_map, file)

            self.display_info(f"Saving done.\n")

            # The dictionary-keys of 'self.smiles_to_graph_map' have been updated in the saved file, thus update the class attribute 'self.smiles_mapped_to_graphs_saved'
            self.smiles_mapped_to_graphs_saved = copy.deepcopy(list(self.smiles_to_graph_map.keys()))

    def _load_measurements_df(self):
        """ Load the measurements (table) as pandas.DataFrame. """
        # Define a dictionary with types of all the columns that are not strings
        dtypes_dict = {
            'pX': float,
            'molecular_weight': float,
            'standard_value (ChEMBL)': float,
            'value (ChEMBL)': float,
            'kinase_listed_on_KinHub': str,
        }

        # Load the measurements
        # Remark: Load all columns as strings and then set the type below for the columns that aren't strings.
        self.display_info('Loading the raw measurements...')
        measurements_df = pd.read_csv(self.measurements_file_path, sep='\t', dtype=str)

        # Specify the type of the columns that are not strings and assign it to the corresponding class attribute
        self.measurements_df = measurements_df.astype(dtypes_dict)

        # Log information
        self.display_info('Loading Done.\n')

    def run(self):
        """ Preprocess the data by first filtering it and then assigning the data to different folds. """
        # Load the map (dictionary) from smiles to graphs for which key-value pairs might be added in the current session using self.smiles_to_graph_mapper
        self.load_smiles_to_graph_map()

        # Filter the raw measurements (generating 'self.filtered_measurements_df')
        self.filter_measurements_df()

        # Generate a list of all non-stereochemical washed canonical SMILES (nswcs) strings of the (filtered) molecules for which
        # no measurements with any kinase are listed on ChEMBL. (generating 'self.non_measured_nswcs_list')
        self.generate_non_measured_nswcs_list()

        # Generate the preprocessed dataset (=> 'self.preprocessed_df')
        self.generate_preprocessed_data_df()

        # Determine the global protein representation parameters
        self.determine_global_protein_representation_params()

        # Generate a map from set (train, valid, test) to the corresponding data fold indices (=> 'self.set_to_data_fold_indices_map')
        self.generate_set_to_data_fold_indices_map()

        # Generate a map from set (train, valid, test) to the corresponding processed subset DataFrame (=> 'self.set_to_processed_subset_df_map')
        self.generate_set_to_processed_subset_df_map()

        # Scramble the processed train subset DataFrame
        self.scramble_processed_train_subset_df()

        # Generate a map from set (train, valid, test) to the corresponding list of 'torch data objects. (=> 'self.set_to_torch_data_list_map')
        self.generate_set_to_torch_data_list_map()

        self.display_info("\nPreprocessing finished.\n")

    def generate_set_to_data_fold_indices_map(self):
        """ Generate a map from set (train, valid, test) to the corresponding (zero-based) data fold indices (as lists). """    
        set_to_data_fold_indices_map = dict()

        # Generate a list of the cross-validation folds for the train-validation set
        cv_fold_indices = list(range(self.K))

        # Generate the data fold indices lists for the train and validation sets depending on the value of k
        if self.k is None:
            # If k is None, no validation set is used, thus the train set contains all cross-validation folds
            # and the validation set contains none.
            set_to_data_fold_indices_map['train'] = cv_fold_indices
            set_to_data_fold_indices_map['valid'] = list()
        else:
            # Otherwise, k is an integer in [0, K-1] (checked when parsing it using 'self.parse_k')
            # Remove k from the list of cross-validation folds (this is done in place)
            cv_fold_indices.remove(self.k)

            # The cross-validation fold indices list doesn't contain the validation data fold index any longer
            # and can thus be assigned to the train set.
            # The validation set contains fold k as only fold.
            set_to_data_fold_indices_map['train'] = cv_fold_indices
            set_to_data_fold_indices_map['valid'] = [self.k]

        # The test set has the largest (zero-based) data fold index corresponding to (#data_folds-1)
        set_to_data_fold_indices_map['test'] = [self.num_data_folds-1]

        # Assign the dictionary to a class attribute of the same name
        self.set_to_data_fold_indices_map = set_to_data_fold_indices_map

    def generate_non_measured_nswcs_list(self):
        """
        Generate a list of molecules for which no measuremenets with kinases as targets are listed on ChEMBL.
        
        Remark: This list will contain the non-stereochemical washed canonical SMILES (nswcs) strings of the molecules.
        """
        ########################################################################################################################################
        ### Step 1: Load relevant files
        ########################################################################################################################################
        # 1) Unique non-stereochemical washed canonical SMILES (nswcs) strings from QMugs summary file
        #    that represent 'all' molecules on ChEMBL
        molecules_from_qmugs_summary_df = pd.read_csv(self.molecules_from_qmugs_summary_file_path, sep='\t')

        # 2) Unique non-stereochemical washed canonical SMILES (nswcs) strings of ALL molecules on ChEMBL
        #    that have been measured for kinases (all data available not only for prefiltered kinases in 
        #    'Activities_Kinases.tsv')
        kinases_chembl_df    = pd.read_csv(self.nswc_smiles_kinases_chembl_file_path, sep='\t')
        kinases_chembl_nswcs = list(kinases_chembl_df['non_stereochemical_washed_canonical_smiles'])

        # 3) Unique non-stereochemical washed canonical SMILES (nswcs) strings of the holdout molecules
        holdout_df    = pd.read_csv(self.nswc_smiles_holdout_file_path, sep='\t')
        holdout_nswcs = list(holdout_df['non_stereochemical_washed_canonical_smiles'])


        ########################################################################################################################################
        ### Step 2: Remove molecules from QMugs summary that have been measured for a kinase (ChEMBL) or are holdout molecules
        ########################################################################################################################################
        # Create a list of molecules (nswcs) that should be removed from the QMugs DataFrame
        # which consists of the holdout molecules (nswcs) and the molecules that have been
        # listed on ChEMBL to been measured for any kinase
        to_be_removed_nswcs = list(set(kinases_chembl_nswcs).union(set(holdout_nswcs)))

        # Filter out all rows that contain these to be removed molecules (nswcs) to obtain
        # a DataFrame containing only molecules that have not been measured for any kinase
        # (and do not correspond to the holdout molecules)
        filtered_df = molecules_from_qmugs_summary_df[~molecules_from_qmugs_summary_df['non_stereochemical_washed_canonical_smiles'].isin(to_be_removed_nswcs)]

        ########################################################################################################################################
        ### Step 3: Loop over these unique molecules and generate their smiles (nswcs) to graph mapping
        ########################################################################################################################################
        self.display_info("Map all of the molecules (i.e. their nswcs) from QMugs (that are neither listed with a kinase on ChEMBL nor holdout molecules) to graphs...")
        unmappable_nswc_smiles_list = list()
        start_time = time.time()
        for nswc_smiles in tqdm.tqdm( set(filtered_df['non_stereochemical_washed_canonical_smiles']) ):
            # Check if the nswc_smiles is already a key of the map
            # Remark: self.smiles_to_graph_map is loaded from a file and thus might already contain this molecule
            if nswc_smiles in self.smiles_to_graph_map:
                continue
            else:
                # Try to map the non-stereochemical washed canonical (nswc) SMILES string to a dictionary describing the molecular graph and
                # assign this graph to the current molecular ChEMBL ID
                try:
                    self.smiles_to_graph_map[nswc_smiles] = self.smiles_to_graph_mapper(nswc_smiles)
                except smiles_to_graph_mapping.MoleculeCannotBeMappedToGraph:
                    # If the error 'smiles_to_graph_mapping.MoleculeCannotBeMappedToGraph' is thrown, if at least one of its 
                    # atoms/vertices or bonds/edges cannot be featurized.
                    # In this case, append the current molecule (nswcs) to the corresponding list and continue to the next molecule.
                    unmappable_nswc_smiles_list.append(nswc_smiles)
                    continue

        self.display_info(f"Finished mapping of all non-measured molecules to graphs. Duration: {time.time()-start_time:.2f}s")
        self.display_info(f"The following {len(unmappable_nswc_smiles_list)} molecules could not be mapped to a graph:\n{unmappable_nswc_smiles_list}\n")

        # Save the smiles to graph map
        self.save_smiles_to_graph_map()

        ########################################################################################################################################
        ### Step 4: Filter the molecules by their molecular weight and then create the sought for list
        ########################################################################################################################################
        # Filter by molecular weight
        non_measured_molecules_df = filtered_df[(self.config_dict['molecular_weight_bounds'][0]<=filtered_df['molecular_weight']) & (filtered_df['molecular_weight']<=self.config_dict['molecular_weight_bounds'][1])]

        # Get the list of the molecules (nswcs) and assign it to the corresponding class attribute
        # Remark: Use list(set(...)) to ensure uniqueness of the molecules
        self.non_measured_nswcs_list = list(set(non_measured_molecules_df['non_stereochemical_washed_canonical_smiles']))

    def filter_measurements_df(self):
        """ Filter the DataFrame (i.e. table) containing the measurements. """
        #####################################################################################################################################################################
        # 1) Remove duplicate entries (=rows in the table)
        #####################################################################################################################################################################
        # This will remove all rows (entries) which are duplicates (so rows whose columns have all exactly the same values)
        filtered_df = self.measurements_df.drop_duplicates()

        # Inform the user how many duplicate entries/rows were removed, if there were any
        num_duplicate_rows = len(self.measurements_df)-len(filtered_df)
        if 0<num_duplicate_rows:
            self.display_info(f"Removed {num_duplicate_rows} rows/entries that were duplicates.")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        # Plot the number of entries per pType
        self.plot_num_entries_per_pType(filtered_df)

        #####################################################################################################################################################################
        # 2) In case that only Kinases listed on KinHub should be used, filter out all entries with kinases NOT listed on KinHub
        #####################################################################################################################################################################
        # Get the boolean flag from the config dictionary that indicates if only kinases listed on KinHub should be used or not
        only_use_kinases_listed_on_KinHub = self.config_dict.get('only_use_kinases_listed_on_KinHub', False)

        # Handle the case where only the kinases listed on KinHub should be used
        if only_use_kinases_listed_on_KinHub:
            # Keep only kinases listed on KinHubList
            # Remark: The entries in the column 'kinase_listed_on_KinHub' are strings equal to 'True' or 'False'
            num_rows_before_filtering = len(filtered_df)
            filtered_df               = filtered_df[filtered_df['kinase_listed_on_KinHub']=='True']

            # Inform the user how many entries/rows were removed in the current filtering step
            num_filtered = num_rows_before_filtering-len(filtered_df)
            self.display_info(f"Removed {num_filtered} rows/entries because their kinase is not listed on KinHub.")

            # Log some general information/summary about the filtered dataset
            self.display_summary(filtered_df)

        #####################################################################################################################################################################
        # 3) Filter out entries with a pX value corresponding to NaN (null in pandas) without an effect comment (=None that is null in pandas)
        #####################################################################################################################################################################
        # Get the number of rows before the filtering and then filter
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[( ~filtered_df['pX'].isnull() ) | ( ~filtered_df['effect_comment'].isnull() )]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because their 'pX value' was NaN, while they had no effect comment.")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)
        #####################################################################################################################################################################

        #####################################################################################################################################################################
        # 4) Filter out non-exact measurements (so keep only entries with 'standard_relation' equal to '=') for all but the entries with an effect_comment (non-null)
        #####################################################################################################################################################################
        # Get the number of rows before the filtering and then filter
        # Remark: Only keep all entries that either have 'relation' equal to '=' or have an effect comment (so it is not null)
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[(filtered_df['relation (ChEMBL)']=='=') | (~filtered_df['effect_comment'].isnull())]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because their measurements were not exact ('relation (ChEMBL)' not equal to '=')\nRemark: Kept all entries with an 'effect_comment' independent of the relation.")
        
        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        # Plot the number of entries per pType
        self.plot_num_entries_per_pType(filtered_df)

        #####################################################################################################################################################################
        # 5) Filter by pType
        #####################################################################################################################################################################
        # Get the number of rows before the filtering and then filter
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[filtered_df['pType'].str.lower().isin(self.config_dict['include_pTypes'])]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because their 'pType' was not in {self.config_dict['include_pTypes']}.")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        #####################################################################################################################################################################
        # 6) Filter by all molecules for which their molecular graph can be generated
        #####################################################################################################################################################################
        # Generate a map from non-stereochemical washed canonical (nswc) SMILES string to molecular graph for all molecules
        # Remark: In case the molecular graph can't be created, there won't be a key-value
        #         pair for this molecule in this map.
        self.display_info("Map all molecules of the filtered dataset to graphs...")
        unmappable_nswc_smiles_list = list()
        start_time = time.time()
        for nswc_smiles in tqdm.tqdm( set(filtered_df['non_stereochemical_washed_canonical_smiles']) ):
            # Check if the nswc_smiles is already a key of the map
            # Remark: self.smiles_to_graph_map is loaded from a file and thus might already contain this molecule
            if nswc_smiles in self.smiles_to_graph_map:
                continue
            else:
                # Try to map the non-stereochemical washed canonical (nswc) SMILES string to a dictionary describing the molecular graph and
                # assign this graph to the current molecular ChEMBL ID
                try:
                    self.smiles_to_graph_map[nswc_smiles] = self.smiles_to_graph_mapper(nswc_smiles)
                except smiles_to_graph_mapping.MoleculeCannotBeMappedToGraph:
                    # If the error 'smiles_to_graph_mapping.MoleculeCannotBeMappedToGraph' is thrown, if at least one of its 
                    # atoms/vertices or bonds/edges cannot be featurized.
                    # In this case, append the current molecule (nswcs) to the corresponding list and continue to the next molecule.
                    unmappable_nswc_smiles_list.append(nswc_smiles)
                    continue

        self.display_info(f"Finished mapping of all molecules of the filtered dataset to graphs. Duration: {time.time()-start_time:.2f}s")
        self.display_info(f"The following {len(unmappable_nswc_smiles_list)} molecules could not be mapped to a graph:\n{unmappable_nswc_smiles_list}\n")

        # Save the smiles to graph map
        self.save_smiles_to_graph_map()

        # Filter out all molecules that could not be mapped to a graph
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[filtered_df['non_stereochemical_washed_canonical_smiles'].isin(self.smiles_to_graph_map.keys())]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because their molecules could not be mapped to a graph.")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        #####################################################################################################################################################################
        # 7) Filter out by the pX values (=> Only keep entries with pX values within the imposed pX boundaries)
        #####################################################################################################################################################################
        # Get the pX boundaries from the config dictionary, returning [None, None] if it is not defined
        pX_bounds = self.config_dict.get('pX_bounds', [None, None])
        
        # In case that one of the boundaries is None, make it either -inf or +inf
        if pX_bounds[0] is None:
            pX_bounds[0] = -np.inf
        if pX_bounds[1] is None:
            pX_bounds[1] = np.inf

        # Get the number of rows before the filtering and then filter
        # Remark: Also do not filter out entries with a pX value corresponding to NaN (null in pandas).
        #         The reason for this is that entries with a pX value corresponding to NaN (or that are labeled)
        #         inactive, are filtered in a previous step (and if they were not filtered before, they should also
        #         not be filtered here).
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[(pX_bounds[0]<=filtered_df['pX']) & (filtered_df['pX']<=pX_bounds[1]) | filtered_df['pX'].isnull()]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because their 'pX value' was not in the interval {pX_bounds}.\nRemark: Did not remove entries with a 'pX value' corresponding to NaN (which can for example be the case for entries with an 'effect_comment').")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)
        #####################################################################################################################################################################

        #####################################################################################################################################################################
        # 8) Filter out entries that are labeled as 'ineffective' in the 'effect_comment'
        #####################################################################################################################################################################
        # Get the boolean flag 'remove_ineffective_labeled_entries' from the config dictionary, returning False if it is not defined
        remove_ineffective_labeled_entries = self.config_dict.get('remove_ineffective_labeled_entries', False)

        # Differ cases where entries labeled as 'ineffective' (via the effect comment) should be removed or not not
        # Remark: 'ineffective' entries have no pX entry. 
        #         Thus, by construction, this should also remove all entries with a pX equal to NaN/None.
        if remove_ineffective_labeled_entries:
            # Get the number of rows before the filtering and then filter
            num_rows_before_filtering = len(filtered_df)
            filtered_df               = filtered_df[filtered_df['effect_comment'].str.lower()!='ineffective']

            # Inform the user how many entries/rows were removed in the current filtering step
            num_filtered = num_rows_before_filtering-len(filtered_df)
            self.display_info(f"Removed {num_filtered} rows/entries because their 'effect_comment' was 'ineffective'.")

            # Log some general information/summary about the filtered dataset
            self.display_summary(filtered_df)
        else:
            # Determine the number of entries with an effect_comment value corresponding to 'ineffective'
            NaN_pX_df = filtered_df[filtered_df['effect_comment'].str.lower()=='ineffective']
            self.display_info(f"Do not remove entries that are labeled 'ineffective' (in the activity data column 'effect_comment') of which there are {len(NaN_pX_df)}.\n")
        #####################################################################################################################################################################

        #####################################################################################################################################################################
        # 9) Filter molecules by their molecular weight
        #####################################################################################################################################################################
        # Remark: Only keep molecules that have a molecular weight within the molecular weight interval boundaries
        # Plot the molecular weights (before filtering)
        self.plot_molecular_weight_histogram(filtered_df['molecular_weight'])

        # Get the number of rows before the filtering and then filter
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[(self.config_dict['molecular_weight_bounds'][0]<=filtered_df['molecular_weight']) & (filtered_df['molecular_weight']<=self.config_dict['molecular_weight_bounds'][1])]

        # Inform the user how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {num_filtered} rows/entries because the weight of their molecules was not in the molecular weight interval {self.config_dict['molecular_weight_bounds']} (units in Daltons).")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        #####################################################################################################################################################################
        # 10) Filter by minimal number of molecules per protein
        #####################################################################################################################################################################
        # Use protein ChEMBL IDs here (old version)
        self.display_info("Determine the number of molecules measured for each protein for the filtered dataset...")
        num_molecules_dict = dict()
        start_time = time.time()
        for protein_id in tqdm.tqdm( set(filtered_df['protein_id']) ):
            # Get all activity entries for the current protein
            protein_df = filtered_df[filtered_df['protein_id']==protein_id]

            # Get the number of molecules tested (after the above filtering) on this protein
            # Remark: This number correspond to the number of unique molecular ChEMBL IDs
            num_molecules_dict[protein_id] = len( set(protein_df['non_stereochemical_washed_canonical_smiles']) )

        self.display_info(f"Finished determination of number of molecules measured for each protein. Duration: {time.time()-start_time:.2f}s\n")

        # Make a plot of the number of proteins vs the threshold of the minimal number of molecules per protein
        self.make_number_of_proteins_plot(num_molecules_dict)

        # Construct a list of protein IDs for proteins that have more molecules than the threshold
        protein_ids = [protein_id for protein_id, num_molecules in num_molecules_dict.items() if self.config_dict['num_molecules_threshold']<=num_molecules]

        # Filter the DataFrame to only keep the proteins in this list
        num_rows_before_filtering = len(filtered_df)
        filtered_df               = filtered_df[filtered_df['protein_id'].isin(protein_ids)]

        # Inform the user how many proteins and also how many entries/rows were removed in the current filtering step
        num_filtered = num_rows_before_filtering-len(filtered_df)
        self.display_info(f"Removed {len(num_molecules_dict)-len(protein_ids)} proteins because their number of measured molecules was less than threshold " \
                          f"of {self.config_dict['num_molecules_threshold']} after the previous filtering steps.")
        self.display_info("Removed {num_filtered} rows/entries because they belonged to one of the removed proteins.")

        # Log some general information/summary about the filtered dataset
        self.display_summary(filtered_df)

        self.display_info("Included protein ids")
        self.display_info("Protein id: #molecules for protein")
        protein_id_mol_num_list = [(protein_id, len(filtered_df[filtered_df['protein_id']==protein_id])) for protein_id in list(set(filtered_df['protein_id']))]
        protein_id_mol_num_list.sort(key=lambda x: x[1], reverse=True)
        for protein_id_mol_num in protein_id_mol_num_list:
            self.display_info(f"{protein_id_mol_num[0]}: {protein_id_mol_num[1]}")
        
        self.display_info('\n\n\n')
        
        # Assign the filtered dataframe 
        self.filtered_measurements_df = filtered_df

    def generate_preprocessed_data_df(self):
        """" Generate the preprocessed data table (i.e., pandas.DataFrame). """
        # Generate the 'measured_df' containing all the (filtered) molecule-protein pairs, which have been measured (ChEMBL), assigned to different data folds.
        measured_df = self.generate_measured_df()

        # Use this to generate the 'non_measured_df' containing all the molecule-protein pairs, which have been measured (QMugs-ChEMBL-Holdout), assigned to different data folds.
        non_measured_df = self.generate_non_measured_df(measured_df)

        # Combine the measured and non-measured DataFrames to 'preprocessed_df'
        # Remark: Using 'ignore_index=True' the old row indices will be ignored and new row indices will be automatically created for the concatenated DataFrame
        preprocessed_df = pd.concat([measured_df, non_measured_df], ignore_index=True)

        # Reorder columns and assign the result to the class attribute of the same name
        col_name_order       = ['on_chembl', 'fold_index', 'protein_id', 'non_stereochemical_washed_canonical_smiles']
        self.preprocessed_df = preprocessed_df[col_name_order]
        self.display_info(f"Number of rows of preprocessed DataFrame (i.e. table): {len(self.preprocessed_df)}.\n")

        #####################################################################################################################
        ### SANITY CHECKS
        #####################################################################################################################
        ####################
        # Ratios:
        ####################
        # Loop over the folds and proteins
        for grouped_values, subset_df in self.preprocessed_df.groupby(by=['fold_index', 'protein_id']):
            fold_index, protein_id = grouped_values
            # Determine the ratio of non-measured (i.e. not on ChEMBL) to measured (i.e. on ChEMBL) entries
            ratio = np.sum(subset_df['on_chembl']==False)/np.sum(subset_df['on_chembl']==True)

        ################################################################################
        # Ensure that there is no overlap between measured and non-measured molecules
        ################################################################################
        # Get the set of molecules (nswcs) for both subsets
        data_measured_nswcs_set     = set(self.preprocessed_df[self.preprocessed_df['on_chembl']==True]['non_stereochemical_washed_canonical_smiles'])
        data_non_measured_nswcs_set = set(self.preprocessed_df[self.preprocessed_df['on_chembl']==False]['non_stereochemical_washed_canonical_smiles'])

        # Get their intersection set and throw an error if it is not empty
        intersection = list(data_measured_nswcs_set.intersection(data_non_measured_nswcs_set))
        if len(intersection)>0:
            err_msg = f"The following molecules are simultaneously in the 'measured' (i.e. on ChEMBL) and 'non-measured' (i.e. not on ChEMBL) subsets: {intersection}"
            raise ValueError(err_msg)

        ################################################################################
        # Ensure that there is no molecule overlap between folds
        ################################################################################
        for fold_index_1 in range(self.num_data_folds):
            for fold_index_2 in range(fold_index_1): 
                # Get the set of molecules (nswcs) for both folds
                data_fold_1_nswcs_set = set(self.preprocessed_df[self.preprocessed_df['fold_index']==fold_index_1]['non_stereochemical_washed_canonical_smiles'])
                data_fold_2_nswcs_set = set(self.preprocessed_df[self.preprocessed_df['fold_index']==fold_index_2]['non_stereochemical_washed_canonical_smiles'])
                
                # Get their intersection set and throw an error if it is not empty
                intersection = list(data_fold_1_nswcs_set.intersection(data_fold_2_nswcs_set))
                if len(intersection)>0:
                    err_msg = f"The following molecules are simultaneously in folds {fold_index_2} & {fold_index_1}: {intersection}"
                    raise ValueError(err_msg)

        self.display_info("Preprocessing done.\n")


    def generate_measured_df(self):  
        """
        Generate a table (i.e. pandas.DataFrame) for all the molecules whose activity to at least one of the kinases (remaining after filtering in 'filter_measurements_df') 
        has been measured (i.e. listed on ChEMBL).
        
        Remark: Attention, do not confuse 'measured_df' with 'measuremends_df' (or its filtered version self.filtered_measurements_df).
                While the former contains molecule-protein pairs (i.e. one entry per pair), the later contains the ChEMBL measurement entries and thus might contain multiple 
                entries for the same protein-molecule pair.

        Return:
            (pandas.DataFrame): Generated 'measured_df'.
            
        """  
        # Assign molecules (nswcs) that have been measured for single proteins to different folds
        fold_index_to_pmp_list_map = self.assign_measured_molecules_to_folds()

        # Create a pandas DataFrame from these
        measured_dict = collections.defaultdict(list)
        for fold_index, pmp_list in fold_index_to_pmp_list_map.items():
            protein_id_list, nswcs_list    = zip(*pmp_list)
            measured_dict['fold_index']   += [fold_index]*len(pmp_list)
            measured_dict['protein_id']   += protein_id_list
            measured_dict['on_chembl']    += [True]*len(pmp_list)
            measured_dict['non_stereochemical_washed_canonical_smiles'] += nswcs_list

        measured_df = pd.DataFrame(measured_dict)

        self.display_info(f"Number of protein-molecule pairs in 'measured_df': {len(measured_df)}\n")

        return measured_df

    def assign_measured_molecules_to_folds(self):
        """
        Assign molecules in the filtered measurements (table) to different data folds
        that have been measured for only one protein to a fold so that the number.

        Return:
            (dict): Dictionary mapping the fold indices to the list of the protein-molecule-pairs
                contained in these fold (without overlap between folds).
        """
        self.display_info(f"Number of entries (filtered measurement table): {len(self.filtered_measurements_df)}\n")
        
        ###############################################################################################################################################################################################
        ### Step 1: Get a list of protein (specified by their protein/ChEMBL ID) and molecule 
        ### (specified by their non-stereochemical washed canonical SMILES strings) tuples
        ###############################################################################################################################################################################################
        protein_id_molecule_nswcs_tuples = list(set(zip(self.filtered_measurements_df['protein_id'], self.filtered_measurements_df['non_stereochemical_washed_canonical_smiles'])))
        protein_id_molecule_nswcs_tuples.sort()
        self.display_info(f"Number of (filtered measured) protein-molecule pairs: {len(protein_id_molecule_nswcs_tuples)}\n")

        ###############################################################################################################################################################################################
        ### Step 2: Assign molecules (nswcs) that have been measured for multiple proteins to different folds
        ###############################################################################################################################################################################################
        nswcs_to_num_proteins_map_multiple, fold_index_to_multiple_nswcs_list_map = self.assign_multiple_target_measured_molecules_to_data_folds(protein_id_molecule_nswcs_tuples)

        ###############################################################################################################################################################################################
        ### Step 3:
        ### Assign molecules in the filtered measurements (table) to different data folds so that the number
        ### of measurements per protein is approximately the same (i.e. stratified) across the folds.
        ###############################################################################################################################################################################################
        # Generate a map from protein ID to non-stereochemical washed canonical SMILES (nswcs) string (i.e. molecules)
        protein_id_to_nswcs_list_map = collections.defaultdict(list)
        for protein_id, nswcs in protein_id_molecule_nswcs_tuples:
            protein_id_to_nswcs_list_map[protein_id].append(nswcs)

        # Get the set of molecules (nswcs) that have been measured for multiple proteins (see above)
        multiple_protein_nswcs_set = set(nswcs_to_num_proteins_map_multiple.keys())

        # Define a new random seed for the single-protein-measurment molecules to the folds
        np.random.seed(self.config_dict['random_seeds']['single_target_measured_molecules_to_data_folds_assignment'])

        # Get a list of the protein IDs, sort these protein IDs (that are the proteins' ChEMBL IDs), and loop over it
        protein_ids_list = list(protein_id_to_nswcs_list_map.keys())
        utils.sort_chembl_id_list(protein_ids_list)
        fold_index_to_pmp_list_map = collections.defaultdict(list)
        for protein_id in protein_ids_list:
            # Get the list of molecules (nswcs) measured for the current protein
            protein_nswcs_list = protein_id_to_nswcs_list_map[protein_id]

            # We want to stratify the number of measurements per protein (that correspond to the number of molecules per protein here)
            # and thus determine the number of molecules per fold for this protein by dividing the total number of molecules for this
            # protein by the number of folds.
            # Remark: By rounding up (ceiling) it can happen that the last fold (i.e. test fold) will contain a little less molecule 
            #         than the others (i.e. in the most unfortunate case '#folds-1' less molecules than the others).
            num_molecules_per_fold = int(np.ceil(len(protein_nswcs_list)/self.num_data_folds))

            # Remove the molecules (nswcs) measured for multiple proteins thereby obtaining a list of molecules (nswcs)
            # that have been only measured for the current protein
            single_protein_nswcs_list = list(set(protein_nswcs_list)-multiple_protein_nswcs_set)

            # Sort and then shuffle the list of the molecules (nswcs) that have only been measured for the current protein
            single_protein_nswcs_list.sort()
            np.random.shuffle(single_protein_nswcs_list)
        
            # Loop over the folds
            num_molecules_required_so_far = 0
            num_molecules_protein = 0
            for fold_index in range(self.num_data_folds):
                # Get the molecules (nswcs) that are already in this fold
                # These have been measured for multiple proteins (and not necessarily the current protein)
                fold_multiple_nswcs_list = fold_index_to_multiple_nswcs_list_map[fold_index]

                # Keep the molecules (nswcs) that have been measured for the current protein
                # Remark: These have also been measured for other proteins
                fold_nswcs_list = list(set(fold_multiple_nswcs_list).intersection(set(protein_nswcs_list)))
                
                # Determine the number of molecules that are already in the current fold for the current protein
                num_molecules_already_in_fold = len(fold_nswcs_list)

                # Determine the difference
                num_molecules_required_for_fold = num_molecules_per_fold-num_molecules_already_in_fold

                # If a negative number of molecules would be required for the current fold, set this number to 0
                if num_molecules_required_for_fold<0:
                    num_molecules_required_for_fold = 0

                # Extract the required number of molecules (nswcs), which have been only measured for the current protein,
                # and add them to the list of molecules (nswcs) of the current fold.
                fold_nswcs_list += single_protein_nswcs_list[num_molecules_required_so_far:num_molecules_required_so_far+num_molecules_required_for_fold]

                # Increase the number of molecules required so far by the number of molecules that were required for the current fold
                num_molecules_required_so_far += num_molecules_required_for_fold

                # Generates a list with 'len(fold_nswcs_list)' elements that all correspond to the current protein ID and then
                # use this list to generate a list of (protein molecule pair) tuples in the form:
                # [(<current_protein_id>, nswcs_1), (<current_protein_id>, nswcs_2), ...]
                # that is assigned (list-addition) to the corresponding dictionary that holds these tuples for each fold.
                # Remark: Sort the molecules (nswcs) list before assigning it.
                fold_protein_list = [protein_id]*len(fold_nswcs_list)
                fold_nswcs_list.sort()
                fold_index_to_pmp_list_map[fold_index] += list(zip(fold_protein_list, fold_nswcs_list))

                # Add the number of molecules of the current fold and protein to the counter for the number of molecules for the entire protein
                num_molecules_protein += len(fold_nswcs_list)

            # Check if the number of molecules assigned to different folds (i.e. num_molecules_protein) for the current protein
            # matches the number of molecules available for the current protein
            if num_molecules_protein!=len(protein_nswcs_list):
                err_msg = f"The number of molecules assigned ({num_molecules_protein}) to different folds for protein {protein_id} does not correspond to the number of available ({len(protein_nswcs_list)}) molecules for this protein! "
                raise ValueError(err_msg)

        return fold_index_to_pmp_list_map

    def assign_multiple_target_measured_molecules_to_data_folds(self, 
                                                                protein_id_molecule_nswcs_tuples):
        """
        Assign molecules that have been measured for multiple protein-targets to a unique fold.
        This will ensure that their datapoints will not appear in multiple folds.
        
        Args:
            protein_id_molecule_nswcs_tuples (list): List containing unique tuples of the form 
                (<protein_id>, <nswcs>) present in the filtered measurements table.

        Return:
            (dict, dict): Two dictionaries: 
                (1) Dictionary mapping NSWCS strings (of molecules measured for multiple protein-targets) 
                    to the number of protein-targets the molecule was measured.
                (2) Dictionary mapping fold indices to the a list to the NSWCS strings (of molecules 
                    measured for multiple protein-targets) contained in these folds (without overlap).

        """
        nswcs_to_protein_id_list_map = collections.defaultdict(list)
        for protein_id, nswcs in protein_id_molecule_nswcs_tuples:
            nswcs_to_protein_id_list_map[nswcs].append(protein_id)

        # Generate a map from molecule (i.e. its nswcs) to the number of proteins the molecule was measured for
        nswcs_to_num_proteins_map = {nswcs: len(protein_id_list) for nswcs, protein_id_list in nswcs_to_protein_id_list_map.items()}

        # Plot the distributions of proteins per molecules (corresponding to the values of 'nswcs_to_num_proteins_map')
        num_proteins_per_molecule = nswcs_to_num_proteins_map.values()

        # Divide the molecules into two dictionaries depending if it was measured for 
        # one (single) or more than one (multiple) proteins
        nswcs_to_num_proteins_map_single   = dict(filter(lambda x: 1==x[1], nswcs_to_num_proteins_map.items()))
        nswcs_to_num_proteins_map_multiple = dict(filter(lambda x: 2<=x[1], nswcs_to_num_proteins_map.items()))

        # Sort the list first by the molecules (nswcs) and then by their weights (in ascending order <=> reverse=True)
        nswcs_num_proteins_tuple_list_multiple = list(nswcs_to_num_proteins_map_multiple.items())
        nswcs_num_proteins_tuple_list_multiple.sort(key=lambda x: x[0])
        nswcs_num_proteins_tuple_list_multiple.sort(key=lambda x: x[1], reverse=True)

        # Get the nswcs (molecules) that were measured for multiple proteins and use their protein numbers as their weights
        nswcs_list, weights_list = zip(*nswcs_num_proteins_tuple_list_multiple)

        # Assign the molecules (nswcs) that have been measured for more than 1 protein to the different folds.
        # This assignment should be such that each fold has molecules with approximately the same amount 
        # (i.e. stratified) of measured proteins (e.g. no all molecules with 40+ proteins end up in fold 0 but
        # they are almost equally distributed over the folds). This is achieved by using a weighted choice of the
        # molecules where the weight (i.e. choice-probability) corresponds to the number of proteins.
        # Remark: Folds includes the cross-validation folds are specified by the number of folds K and the fold 
        #        index k in [0, ..., K-1] comprising the train-validation set and the test set fold (i.e. k=K)
        np.random.seed(self.config_dict['random_seeds']['multiple_target_measured_molecules_to_data_folds_assignment'])

        # Determine the number of choices
        # Remark: Rounding up (ceiling) means that in the last choice not all folds will get molecules assigned
        num_choices = int(np.ceil(len(nswcs_list)/self.num_data_folds))

        # Loop over the choices
        fold_index_to_multiple_nswcs_list_map = collections.defaultdict(list)
        for choice_index in range(num_choices):
            # Get the nswcs (sorted by primarily by weight and secondarily by their alpha-numeric name) 
            # for the current choice
            choice_nswcs_list   = nswcs_list[choice_index*self.num_data_folds:(choice_index+1)*self.num_data_folds]

            # Randomly choose the molecules without replacement and with the probability corresponding to their protein number
            # Remark: By doing this #molecules times we obtain a shuffled nswcs array where the order corresponds to the choosing order
            #         (i.e. more probable items are expected at the beginning and less probable ones at the end)
            choice_fold_indices = np.random.choice(list(range(self.num_data_folds)), size=self.num_data_folds, replace=False)
            
            # Loop over the chosen fold indices for each of the molecules (nswcs) of the current choice and
            # assign them to the corresponding fold
            for item_index, fold_index in enumerate(choice_fold_indices):
                # As we are rounding up when determining the number of choices above, the last choice might
                # not have enough molecules (nswcs) for all the folds
                if item_index<len(choice_nswcs_list):
                    fold_index_to_multiple_nswcs_list_map[fold_index].append(choice_nswcs_list[item_index])

        return nswcs_to_num_proteins_map_multiple, fold_index_to_multiple_nswcs_list_map

    def generate_non_measured_df(self, 
                                 measured_df):
        """ 
        Generate the table holding all non-measured molecules (i.e. molecules from QMugs summary whose activity has not been measured for any kinase on ChEMBL). 

        Args:
            measured_df (pandas.DataFrame): DataFrame holding the molecules (and associated protein-targets) 
                that have been measured.

        Return:
            (pandas.DataFrame): DataFrame corresponding to the generated table holding all non-measured molecules.
        
        """
        # Extract parameters for subsampling from the relevant parameters dictionary
        if self.config_dict['non_measured_molecules_subsampling_params'] is None:
            non_measured_molecules_subsampling_flag    = False
        else:
            non_measured_molecules_subsampling_flag    = True
            ratio_non_measured_to_measured_subsampling = self.config_dict['non_measured_molecules_subsampling_params'].get('ratio_non_measured_to_measured', 1)
            segment_index_non_measured_subsampling     = self.config_dict['non_measured_molecules_subsampling_params'].get('segment_index', 0)

        # Set a random seed for the assignment of non-measured molecule to the folds
        np.random.seed(self.config_dict['random_seeds']['non_measured_molecules_to_data_folds_assignment'])

        # Sort and shuffle the non-measured molecules [i.e. their non-stereochemical washed canonical SMILES (nswcs) strings]
        self.non_measured_nswcs_list.sort()
        np.random.shuffle(self.non_measured_nswcs_list)

        # Determine the ratio of non-measured to measured "entries"
        ratio_non_measured_to_measured = len(self.non_measured_nswcs_list)/len(measured_df)

        # In case that the non-measured molecules should be subsampled, do some sanity checks
        if non_measured_molecules_subsampling_flag:
            # In case that the ratio of non-measured to measured is bigger than the subsampling ratio, throw an error
            if ratio_non_measured_to_measured<ratio_non_measured_to_measured_subsampling:
                err_msg = f"The requested non-measurement subsampling ration ({ratio_non_measured_to_measured_subsampling}) is larger than the ratio of non-measured to measured ({ratio_non_measured_to_measured}).\nPlease select a smaller ratio."
                raise ValueError(err_msg)

            # Determine the number of (non-overlapping) subsampling segments
            # Explanation: Assume we have the shuffled molecules list with elements
            #             [m_1, m_2, m_3, m_4, m_5, m_6]
            #             if we subsample this list to 2 elements, we have 3 non-overlapping segement
            #             [m_1, m_2], [m_3, m4], [m_5, m_6]
            num_segments_non_measured_subsampling = int(np.floor(ratio_non_measured_to_measured/ratio_non_measured_to_measured_subsampling))

            # Check that the segment index (zero based!) does not exceed the number of segments
            if num_segments_non_measured_subsampling<=segment_index_non_measured_subsampling:
                err_msg = f"The zero-based segment index ({segment_index_non_measured_subsampling}) for the non-measured molecule subsampling is larger than (or equal to) the number of segments ({num_segments_non_measured_subsampling}).\nPlease select a smaller segement index."
                raise ValueError(err_msg)

        # Get a list of all protein IDs (that are ChEMBL IDs) and sort them 
        protein_id_list = list(set(measured_df['protein_id']))
        utils.sort_chembl_id_list(protein_id_list)

        # Set a random seed for the assignment of non-measured molecule to the proteins (within folds)
        np.random.seed(self.config_dict['random_seeds']['non_measured_molecules_to_proteins_assignment'])

        # Loop over the folds
        num_non_measured_molecules_extracted = 0
        non_measured_dict = collections.defaultdict(list)
        for fold_index in range(self.num_data_folds):
            # Get the sub-DataFrame of the measured DataFrame for the current fold
            fold_measured_df = measured_df[measured_df['fold_index']==fold_index]

            # Loop over the proteins
            for protein_id in protein_id_list:
                # Get the number of measured molecules that in the data for the current protein and fold (pf)
                pf_num_measured_molecules = len(fold_measured_df[fold_measured_df['protein_id']==protein_id])
                
                # Determine the number of non-measured molcules that should be extracted for the current protein and fold (pf)
                pf_num_non_measured_molecules = int(np.floor(ratio_non_measured_to_measured*pf_num_measured_molecules))
                
                # Extract this number of molecules from the list of non-measured molecules (nswcs) for the current protein and fold (pf)
                pf_non_measured_nswcs_list = self.non_measured_nswcs_list[num_non_measured_molecules_extracted:num_non_measured_molecules_extracted+pf_num_non_measured_molecules]

                # Add the number of extracted non-measured molecules for the current protein and fold (pf) to the global 
                # molecule assignment counter 'num_non_measured_molecules_extracted'
                num_non_measured_molecules_extracted += pf_num_non_measured_molecules        
                
                ##########################################################################################################################
                # Remark: We always extract the same molecules for the protein and fold (only dependent on shuffling random seed).
                #         Using subsampling, only a subset of these extracted molecules might then be used at a time.
                ##########################################################################################################################

                # In case the non-measured molecules should be subsampled
                if non_measured_molecules_subsampling_flag:
                    # Determine the number of the subsampled molecules
                    pf_num_non_measured_molecules_subsampling = int(np.floor(ratio_non_measured_to_measured_subsampling*pf_num_measured_molecules))

                    # Update the non-measured molecule (nswcs) list using only the molecules for the selected segment
                    start_index = segment_index_non_measured_subsampling*pf_num_non_measured_molecules_subsampling
                    end_index   = (segment_index_non_measured_subsampling+1)*pf_num_non_measured_molecules_subsampling
                    pf_non_measured_nswcs_list = pf_non_measured_nswcs_list[start_index:end_index]
                    
                # Sort the list of non-measured molecules (nswcs) for the current protein and fold (pf)
                pf_non_measured_nswcs_list.sort()
        
                # Append all quantities to their corresponding lists in the the non-measured dictionary
                non_measured_dict['fold_index'] += [fold_index]*len(pf_non_measured_nswcs_list)
                non_measured_dict['protein_id'] += [protein_id]*len(pf_non_measured_nswcs_list)
                non_measured_dict['on_chembl']  += [False]*len(pf_non_measured_nswcs_list)
                non_measured_dict['non_stereochemical_washed_canonical_smiles'] += pf_non_measured_nswcs_list

            # In case the fold is the last one, assign all remaining non-measured molecules to the proteins of this fold
            # Remark: If the non-measured molecules should be subsampled, also skip the following
            if fold_index==(self.num_data_folds-1) and not non_measured_molecules_subsampling_flag:
                # Extract the remaining (i.e. not already extracted) non-measured molecules
                # Determine the number of remaining molecules (nswcs)
                remaining_non_measured_molecules = self.non_measured_nswcs_list[num_non_measured_molecules_extracted:]

                # Generate a list of number of measured molecules per protein (for the current fold)
                fold_num_measured_molecules_per_protein_list = [len(fold_measured_df[fold_measured_df['protein_id']==protein_id]) for protein_id in protein_id_list]

                # Randomly assign non-measured molecules to proteins using the number of measured molecules for each protein as weight (for the current fold)
                # 1) Generate probabilities from the weights (that correspond to the number of measured molecules for each protein in the current fold)
                p = np.array(fold_num_measured_molecules_per_protein_list)/np.sum(fold_num_measured_molecules_per_protein_list)

                # 2) Use numpy random choice to get the assignment (index of the non-measured molecules list -> protein ID)
                # Illustration: Assume we have two protein IDs [protein_id_1, protein_id_2] with weights [1, 2], and the non-measured molecules:
                #              [molecule_1, molecule_2, molecule_3, molecule_4, molecule_5].
                #              The protein assignment will then be randomly chosen such as for example:
                #              [protein_id_2, protein_id_1, protein_id_2, protein_id_2, protein_id_1]
                #              Which is equivalent to the assignment:
                #              protein_id_1 <- [molecule_2, molecule_5]
                #              protein_id_2 <- [molecule_1, molecule_3, molecule_4]}
                protein_assignment = np.random.choice(protein_id_list, size=len(remaining_non_measured_molecules), replace=True, p=p)
                
                # 3) Transform the protein assignment to a map from protein ID to a list of assigned on-measured molecules (nswcs) the current fold
                protein_id_to_nswcs_list_map = collections.defaultdict(list)
                for item_index, protein_id in enumerate(protein_assignment):
                    # Get the current molecule assigned to the current protein and current fold (pf)
                    # Remark: See illustration of the assignment above [in (2)] for an explanation
                    nswcs_list = remaining_non_measured_molecules[item_index]

                    # Append the current molecule (nswcs) to the corresponding protein list (in the dictionary)
                    protein_id_to_nswcs_list_map[protein_id].append(nswcs_list)

                # 4) Loop over all proteins and their molecules to the global dictionary (of lists) 'non_measured_dict'
                for protein_id in protein_id_list:
                    # Get the list of molecules (nswcs) assigned to the current protein (for the current fold) and sort it
                    nswcs_list = protein_id_to_nswcs_list_map[protein_id]
                    nswcs_list.sort()

                    # Append all quantities to their corresponding lists in the the non-measured dictionary
                    non_measured_dict['fold_index'] += [fold_index]*len(nswcs_list)
                    non_measured_dict['protein_id'] += [protein_id]*len(nswcs_list)
                    non_measured_dict['on_chembl']  += [False]*len(nswcs_list)
                    non_measured_dict['non_stereochemical_washed_canonical_smiles'] += nswcs_list


        # Transform 'non_measured_dict' to a pandas DataFrame 
        non_measured_df = pd.DataFrame(non_measured_dict)

        self.display_info(f"Number of total non-measured (filtered) molecules: {len(self.non_measured_nswcs_list)}\n")
        self.display_info(f"Number of rows in the (subsampled) non-measured DataFrame: {len(non_measured_df)}\n")

        return non_measured_df

    def determine_global_protein_representation_params(self):
        """ Determine the global protein representation parameters. """
        # Initialize self.protein_representation_params as dictionary containing the protein representation type
        self.protein_representation_params = {'type': self.config_dict['protein_representation_type']}

        # Differ cases based on the protein representation type
        if self.config_dict['protein_representation_type']=='protein_sequence':
            raise NotImplementedError("Protein sequence featurization has not been implemented.")
        elif self.config_dict['protein_representation_type']=='protein_index':
            # Index the proteins
            # First, construct a list of all protein IDs in the preprocessed data
            protein_id_list = list( set(self.preprocessed_df['protein_id']) )
            
            # Second, sort the list of protein ChEMBL IDs
            utils.sort_chembl_id_list(protein_id_list)
            
            # Third, generate a mapping from protein ChEMBL ID to protein index
            self.protein_representation_params['protein_id_to_protein_index_map'] = {protein_id: protein_index for protein_index, protein_id in enumerate(protein_id_list)}
        else:
            err_msg = f"The variable 'protein_representation_type' must be either 'protein_sequence' or 'protein_index', got value '{self.config_dict['protein_representation_type']}' instead."
            raise ValueError(err_msg)
        
    def generate_set_to_processed_subset_df_map(self):
        """ Generate a map from set (train, valid, test) to the corresponding processed subset DataFrame. """
        # Information to the user
        self.display_info(f"Generate a map from set (train, valid, test) to the corresponding processed subset DataFrame.\n")

        # Loop over the (sub)set names and corresponding data fold indices (list) that are the key-value pairs of 
        # self.set_to_data_fold_indices_map
        set_to_processed_subset_df_map = dict()
        for set_name, fold_indices in self.set_to_data_fold_indices_map.items():
            # Get the row indices of the preprocessed DataFrame (i.e. table) for the data folds corresponding to the current set
            set_row_indices = list(self.preprocessed_df.index[self.preprocessed_df['fold_index'].isin(fold_indices)])

            # Sort these indices and use them to generate a subset DataFrame (i.e. table) of the preprocessed DataFrame (i.e. table) for the current set
            # Remark: We use index slicing (on sorted indices) here to ensure a reproducible row order in the subset DataFrame
            set_row_indices.sort()
            set_to_processed_subset_df_map[set_name] = self.preprocessed_df.loc[set_row_indices]

        # Assign 'set_to_processed_subset_df_map' to the corresponding class attribute
        self.set_to_processed_subset_df_map = set_to_processed_subset_df_map

    def scramble_processed_train_subset_df(self):
        """ [Wrapper] Scramble the connections between protein and molecules (i.e. the protein-molecule pairs). """
        if 'train_data_scrambling' in self.config_dict:
            scrambling_strategy = self.config_dict['train_data_scrambling'].get('strategy', 'swap_connections')
            self.display_info(f"The scrambling strategy is: '{scrambling_strategy}'")
            if scrambling_strategy=='swap_connections':
                scrambled_connections_map = self.get_scrambled_connections_map_for_train_subset_df_sc()
            elif scrambling_strategy=='sample_non_measured_as_positives':
                scrambled_connections_map = self.get_scrambled_connections_map_for_train_subset_df_snmap()
            else:
                err_msg = f"The scrambling strategy must be either 'swap_connections' or 'sample_non_measured_as_positives'"
                raise ValueError(err_msg)
        else:
            # If no 'train_data_scrambling' is specified in the configuration dictionary, do not scramble
            # the (train) data and thus use an empty 'scrambled_connections_map' in the following
            scrambled_connections_map = dict()

            # Inform the user of this case
            self.display_info("'train_data_scrambling' is not specified in config_dict and thus no data-scrambling is applied.")

        # Get the scrambled (preprocessed) train (sub)set DataFrame
        scrambled_train_set_df = self.get_scrambled_train_set_df(scrambled_connections_map)

        # Display information to the user
        original_connection_set = set(zip(scrambled_train_set_df['protein_id'], scrambled_train_set_df['original_non_stereochemical_washed_canonical_smiles']))
        adjusted_connection_set = set(zip(scrambled_train_set_df['protein_id'], scrambled_train_set_df['non_stereochemical_washed_canonical_smiles']))
        intersection_set = adjusted_connection_set.intersection(original_connection_set)
        total_scrambling_fraction = 1-len(intersection_set)/len(original_connection_set)
        self.display_info(f"Sanity check (with 'scrambled_train_set_df'):\nScrambled fraction (over all connections): {total_scrambling_fraction*100:.3f}%\n")

        # Update the (preprocessed) train (sub)set DataFrame with the scrambled version
        self.set_to_processed_subset_df_map['train'] = scrambled_train_set_df

    def get_scrambled_connections_map_for_train_subset_df_snmap(self):
        """
        Scramble by 'sampling non-measured molecules' as positives.

        Return a dictionary mapping original connections (i.e. protein-molecule pairs) 
        to scrambled connections using the original connectivity of the (preprocessed)
        train (sub)set.
        
        Return:
            (dict): Dictionary mapping original connections (i.e. protein-molecule
                pairs) of the form (protein_id, nswcs) to their corresponding scrambled 
                connection that will have the same protein_id but a different (i.e. scrambled) 
                nswcs.
        
        """
        # If any except the 'on_chembl' connections should be scrambled, throw an error
        if self.config_dict['train_data_scrambling']['which_connections']!='on_chembl':
            err_msg = f"Cannot scramble connections that are not 'on_chembl' connections"
            raise ValueError(err_msg)
        
        # Define a new random seed for the scrambling
        np.random.seed(self.config_dict['random_seeds']['train_data_scrambling'])

        # Get the (preprocessed) train (sub)set DataFrame
        train_set_df = self.set_to_processed_subset_df_map['train']

        # Get all nswcs that are listed on ChEMBL
        # interactions (i.e. connections) with proteins
        on_chembl_train_set_df = train_set_df[train_set_df['on_chembl']==True]

        # Determine all connections (i.e. protein-molecule pairs) that are listed on ChEMBL
        on_chembl_connections_set = set(zip(on_chembl_train_set_df['protein_id'], on_chembl_train_set_df['non_stereochemical_washed_canonical_smiles']))

        # Get a (sorted) list of protein IDs
        protein_id_list = list(set(list(train_set_df['protein_id'])))
        protein_id_list.sort()

        # Generate two dictionaries:
        # (1) Mapping protein IDs (i.e. proteins) to a list of all nswcs (i.e. molecules) they are connected to
        # (2) Mapping nswcs (i.e. molecules) to a list of all protein IDs (i.e. proteins) they are connected to
        # only involving the connections listed on ChEMBL.
        protein_id_to_nswcs_list_map = {protein_id: list(set(list(on_chembl_train_set_df[on_chembl_train_set_df['protein_id']==protein_id]['non_stereochemical_washed_canonical_smiles']))) for protein_id in protein_id_list}

        # Get a all non-measured nswcs that were sampled as 'negatives' for the train, validation, and test sets
        sampled_non_measured_nswcs_list = list()
        for subset_df in self.set_to_processed_subset_df_map.values():
            non_measured_subset_df = subset_df[subset_df['on_chembl']==False]
            sampled_non_measured_nswcs_list += list(non_measured_subset_df['non_stereochemical_washed_canonical_smiles'])
            
        # Get all non-measured nswcs that were NOT sampled as 'negatives' for the train, validation, and test sets.
        # Remark: self.non_measured_nswcs_list is the list of all preprocessed/filtered nswcs 
        unsampled_non_measured_nswcs_set = set(self.non_measured_nswcs_list)-set(sampled_non_measured_nswcs_list)
        
        scrambled_connections_map = dict()
        for protein_id in protein_id_list:
            # Get the list of nswcs measured for the current protein
            nswcs_list = list(set(protein_id_to_nswcs_list_map[protein_id]))
            nswcs_list.sort()

            # For each of these nswcs (i.e. molecule) sample a new nswcs 
            # from the so-far unsampled non-measured set of nswcs
            candidate_nswcs_list = list(unsampled_non_measured_nswcs_set)
            candidate_nswcs_list.sort()
            sampled_nswcs_list   = list(np.random.choice(candidate_nswcs_list, size=len(nswcs_list), replace=False))

            # Generate a map from original connection to scrambled connection that have
            # the form (protein_id, nswcs) and (protein_id, sampled_nswcs), respectively,
            # for the current protein. 
            # Use it to update the corresponding map over all proteins.
            for nswcs, sampled_nswcs in zip(nswcs_list, sampled_nswcs_list):
                # Scramble a connection with probability p_{scrambling} (i.e. mapping
                # a connection [dictionary-key] to the corresponding scrambled connection 
                # [dictionary-value]).
                u = np.random.uniform(0, 1)
                if u<=self.config_dict['train_data_scrambling']['p_scrambling']:
                    scrambled_connections_map[(protein_id, nswcs)] = (protein_id, sampled_nswcs)

            # Remove the sampled nswcs from the set of so-far unsampled non-measured nswcs
            unsampled_non_measured_nswcs_set = unsampled_non_measured_nswcs_set - set(sampled_nswcs_list)

        # Sanity checks
        scrambled_connections = set(scrambled_connections_map.values())
        scrambled_fraction    = len(scrambled_connections)/len(on_chembl_connections_set)
        self.display_info(f"Sanity check (with 'scrambled_connections_map'):\nScrambled fraction (only over ChEMBL connections): {scrambled_fraction*100:.3f}%")

        return scrambled_connections_map

    def get_scrambled_connections_map_for_train_subset_df_sc(self):
        """
        Scramble by 'swapping connections'.

        Return a dictionary mapping original connections (i.e. protein-molecule pairs) 
        to scrambled connections using the original connectivity of the (preprocessed)
        train (sub)set.
        
        Return:
            (dict): Dictionary mapping original connections (i.e. protein-molecule
                pairs) of the form (protein_id, nswcs) to their corresponding scrambled 
                connection that will have the same protein_id but a different (i.e. scrambled) 
                nswcs.
        
        """
        # Define a new random seed for the scrambling
        np.random.seed(self.config_dict['random_seeds']['train_data_scrambling'])

        # Get the (preprocessed) train (sub)set DataFrame
        train_set_df = self.set_to_processed_subset_df_map['train']

        # Get all protein IDs
        protein_ids = list(set(list(train_set_df['protein_id'])))

        # Differ the cases where only the connections (i.e. protein-molecule pairs) listed on
        # ChEMBL or ALL connections should be scrambled.
        if self.config_dict['train_data_scrambling']['which_connections']=='on_chembl':
            ### Scramble only connections listed on ChEMBL
            # Get all nswcs that are listed on ChEMBL
            # interactions (i.e. connections) with proteins
            on_chembl_train_set_df  = train_set_df[train_set_df['on_chembl']==True]
            on_chembl_nswcs_set = set(list(on_chembl_train_set_df['non_stereochemical_washed_canonical_smiles']))

            # Determine all connections (i.e. protein-molecule pairs) that are listed on ChEMBL
            all_connections_list = list(zip(on_chembl_train_set_df['protein_id'], on_chembl_train_set_df['non_stereochemical_washed_canonical_smiles']))

            # Generate two dictionaries:
            # (1) Mapping protein IDs (i.e. proteins) to a list of all nswcs (i.e. molecules) they are connected to
            # (2) Mapping nswcs (i.e. molecules) to a list of all protein IDs (i.e. proteins) they are connected to
            # only involving the connections listed on ChEMBL.
            protein_id_to_nswcs_list_map = {protein_id: list(set(list(on_chembl_train_set_df[on_chembl_train_set_df['protein_id']==protein_id]['non_stereochemical_washed_canonical_smiles']))) for protein_id in protein_ids}
            nswcs_to_protein_id_list_map = {nswcs: list(set(list(on_chembl_train_set_df[on_chembl_train_set_df['non_stereochemical_washed_canonical_smiles']==nswcs]['protein_id']))) for nswcs in on_chembl_nswcs_set}
        elif self.config_dict['train_data_scrambling']['which_connections']=='all':
            ### Scramble all connections
            # Get all nswcs
            all_nswcs_set = set(list(train_set_df['non_stereochemical_washed_canonical_smiles']))

            # Determine all connection (i.e. protein-molecule pairs)
            all_connections_list = list(zip(train_set_df['protein_id'], train_set_df['non_stereochemical_washed_canonical_smiles']))

            # Generate two dictionaries:
            # (1) Mapping protein IDs (i.e. proteins) to a list of all nswcs (i.e. molecules) they are connected to
            # (2) Mapping nswcs (i.e. molecules) to a list of all protein IDs (i.e. proteins) they are connected to
            protein_id_to_nswcs_list_map = {protein_id: list(set(list(train_set_df[train_set_df['protein_id']==protein_id]['non_stereochemical_washed_canonical_smiles']))) for protein_id in protein_ids}
            nswcs_to_protein_id_list_map = {nswcs: list(set(list(train_set_df[train_set_df['non_stereochemical_washed_canonical_smiles']==nswcs]['protein_id']))) for nswcs in all_nswcs_set}
        else:
            err_msg = f"The configuration 'train_data_scrambling.which_connections' must be either 'on_chembl' or 'all', got '{self.config_dict['train_data_scrambling']['which_connections']}' instead."
            raise ValueError(err_msg)

        # Scramble to molecule-protein connections
        scrambled_connections_map = self.get_scrambled_connections_map(nswcs_to_protein_id_list_map, protein_id_to_nswcs_list_map)

        #####################################################################################
        ### Do sanity checks
        #####################################################################################
        original_connections = set(scrambled_connections_map.keys())
        scrambled_connections = set(scrambled_connections_map.values())
        intersection_set = original_connections.intersection(scrambled_connections)
        if 0<len(intersection_set):
            err_msg = f"There was an overlap between the original and scrambled connections."
            raise ValueError(err_msg)
        
        # Extra checks in the case of self.config_dict['train_data_scrambling']['p_scrambling']=1
        if self.config_dict['train_data_scrambling']['p_scrambling']==1:
            if original_connections!=set(all_connections_list):
                err_msg = f"In case that p_scrambling=1, the set of original connections must match the set of all actual connections, but it did not."
                raise ValueError(err_msg)
            
            if set(nswcs_to_protein_id_list_map.keys())!=set([connection[1] for connection in scrambled_connections_map.values()]):
                err_msg = f"In case that p_scrambling=1, all nswcs must appear in the scrambled connections, but they did not."
                raise ValueError(err_msg)

        scrambled_fraction = len(scrambled_connections)/len(all_connections_list)
        if self.config_dict['train_data_scrambling']['which_connections']=='on_chembl':
            remark = '(only over ChEMBL connections)'
        else:
            remark = '(over all connections)'
        self.display_info(f"Sanity check (with 'scrambled_connections_map'):\nScrambled fraction {remark}: {scrambled_fraction*100:.3f}%")
        #####################################################################################

        return scrambled_connections_map
    
    def get_scrambled_connections_map(self, 
                                      nswcs_to_protein_id_list_map, 
                                      protein_id_to_nswcs_list_map):
        """
        Return a dictionary mapping original connections (i.e. protein-molecule pairs) 
        to scrambled connections.
        

        Args:
            nswcs_to_protein_id_list_map (dict): Dictionary mapping each nswcs to a list of the
                protein IDs that they have been measured on (within a certain data subset).
            protein_id_to_nswcs_list_map (dict): Dictionary mapping each protein ID to a list of 
                the nswcs that they have been measured on (within a certain data subset).

        Return:
            (dict): Dictionary mapping original connections (i.e. protein-molecule
                pairs) of the form (protein_id, nswcs) to their corresponding scrambled 
                connection that will have the same protein_id but a different (i.e. scrambled) 
                nswcs.

        """
        # Determine all nswcs and protein IDs
        all_nswcs_set      = set(list(nswcs_to_protein_id_list_map.keys()))
        all_protein_id_set = set(list(protein_id_to_nswcs_list_map.keys()))

        # Determine maps for 
        # (1) nswcs to a list of its connections of the form '(protein_id, nswcs)'
        # (2) protein IDs to a list of its connections of the form '(protein_id, nswcs)'
        nswcs_to_connection_list_map      = {nswcs: list(zip(nswcs_to_protein_id_list_map[nswcs], [nswcs]*len(nswcs_to_protein_id_list_map[nswcs]))) for nswcs in nswcs_to_protein_id_list_map}
        protein_id_to_connection_list_map = {protein_id: list(zip([protein_id]*len(protein_id_to_nswcs_list_map[protein_id]), protein_id_to_nswcs_list_map[protein_id])) for protein_id in protein_id_to_nswcs_list_map}

        # Determine maps from nswcs/protein IDs to number of connections they have, 
        # i.e. for how many proteins/nswcs they have been measured( within the data)
        nswcs_to_num_connections_map = {nswcs: len(connection_list) for nswcs, connection_list in nswcs_to_connection_list_map.items()}
        protein_id_to_num_connections_map = {protein_id: len(connection_list) for protein_id, connection_list in protein_id_to_connection_list_map.items()}
        
        # Define 'global' variables (i.e. appearing in both steps)
        nswcs_to_num_picks_map      = dict()
        removed_nswcs_set           = set()
        removed_protein_id_set      = set()
        protein_id_to_num_picks_map = dict()
        scrambled_connection_set    = set()

        ################################################################################
        # Step 1: Loop over all nswcs once and assign each of them to a single protein
        #         This ensures, that all nswcs occur in the scrambled dataset.
        ################################################################################
        nswcs_list = list(set(nswcs_to_connection_list_map.keys()))
        nswcs_list.sort()              # In-place
        np.random.shuffle(nswcs_list)  # In-place
        for nswcs in nswcs_list:
            # Get the list of protein IDs measured for the current nswcs
            protein_id_list = nswcs_to_protein_id_list_map[nswcs]

            # Determine the list of all proteins that have not been measured for
            # the current protein (and thus are not connected to it) that is
            # given by all protein IDs without the ones observed for the current
            # protein or the proteins that are already full (and thus removed)
            # These are all candidates for the scrambled protein assignment.
            candidate_protein_id_list = list(all_protein_id_set - set(protein_id_list) - removed_protein_id_set)
            candidate_protein_id_list.sort()  # In-place

            # Get the number of proteins each of the original nswcs is connected to
            num_connections_candidate_protein_id_list = [protein_id_to_num_connections_map[protein_id] for protein_id in candidate_protein_id_list]

            # Sample (without replace) nswcs that have not been measured for the current
            # protein of a total number equal to the number of nswcs in the original data
            # Remark: Use the number of connections per nswcs candidate to determine the
            #         'sampling-probability/weight' of each nswcs candidate.
            candidate_protein_id_probs = np.array(num_connections_candidate_protein_id_list)/np.sum(num_connections_candidate_protein_id_list)
            sampled_protein_id = list( np.random.choice(np.array(candidate_protein_id_list), 1, replace=False, p=candidate_protein_id_probs) )[0]
            scrambled_connection_set.add((sampled_protein_id, nswcs))

            # Increase the number of times the protein ID was picked (i.e. sampled) by 1
            if sampled_protein_id not in protein_id_to_num_picks_map:
                protein_id_to_num_picks_map[sampled_protein_id] = 1
            else:
                protein_id_to_num_picks_map[sampled_protein_id] += 1

            # If the number of picks matches or exceeds the real number of connections of a protein,
            # the protein is full and thus should be removed (from future assignments).
            if protein_id_to_num_connections_map[sampled_protein_id]<=protein_id_to_num_picks_map[sampled_protein_id]:
                removed_protein_id_set.add(sampled_protein_id)

            # Increase the number of times the nswcs was picked (i.e. assigned) by 1
            if nswcs not in nswcs_to_num_picks_map:
                nswcs_to_num_picks_map[nswcs] = 1
            else:
                # The nswcs should not have been picked before, so this case
                # here should actually not happen. Throw an error if it did.
                raise ValueError(f"The current nswcs\n{nswcs}\nhas already been picked before although this should be the first time that it could have been picked.")

            # If the number of picks matches or exceeds the real number of connections of a nswcs,
            # the nswcs assignment is exhausted and thus should be removed (from future assignments).
            if nswcs_to_num_connections_map[nswcs]<=nswcs_to_num_picks_map[nswcs]:
                removed_nswcs_set.add(nswcs)

        # Create a map from protein ID to list of already scrambled nswcs
        protein_id_to_scrambled_nswcs_list_map = collections.defaultdict(list)
        for connection in scrambled_connection_set:
            protein_id_to_scrambled_nswcs_list_map[connection[0]].append(connection[1])

        ##################################################################################
        # Step 2: Loop over all protein IDs and assign nswcs to them until they are full
        ##################################################################################
        # Determine a list of tuples (protein_id, #connections) and sort this
        # list in increasing order in the number of connections
        protein_id_num_connections_tuple_list = [(protein_id, len(connection_list)) for protein_id, connection_list in protein_id_to_connection_list_map.items()]
        protein_id_num_connections_tuple_list.sort(key=lambda x: x[1])  # In-place
        sorted_protein_id_list = [item[0] for item in protein_id_num_connections_tuple_list]
        for protein_id in sorted_protein_id_list:
            # Get the list of nswcs measured for the current protein
            nswcs_list = protein_id_to_nswcs_list_map[protein_id]

            # Get the set of already scrambled nswcs for the current protein
            already_scrambled_nswcs_set = set(protein_id_to_scrambled_nswcs_list_map[protein_id])

            # Determine the list of all nswcs that have not been measured for
            # the current protein (and thus are not connected to it) that is
            # given by all nswcs without the ones observed for the current
            # protein or the nswcs that have already been scrambled or removed.
            # These are all candidates for the scrambled molecule assignment.
            candidate_nswcs_list = list(all_nswcs_set - set(nswcs_list) - removed_nswcs_set - already_scrambled_nswcs_set)
            candidate_nswcs_list.sort()  # In-place

            # Determine the number of nswcs measured for the current protein
            # (within the dataset) minus the number of nswcs already picked
            # for the current protein
            num_samples = len(nswcs_list)-protein_id_to_num_picks_map[protein_id]

            # Check that this number is smaller than the number of candidate nswcs
            if len(candidate_nswcs_list)<num_samples:
                err_msg = f"There are not enough non-connected nswcs available for protein {protein_id}."
                raise ValueError(err_msg)

            # Get the number of proteins each of the original nswcs is connected to
            num_connections_candidate_nswcs_list = [nswcs_to_num_connections_map[nswcs] for nswcs in candidate_nswcs_list]

            # Sample (without replace) nswcs that have not been measured for the current
            # protein of a total number equal to the number of nswcs in the original data
            # Remark: Use the number of connections per nswcs candidate to determine the
            #         'sampling-probability/weight' of each nswcs candidate.
            candidate_nswcs_probs = np.array(num_connections_candidate_nswcs_list)/np.sum(num_connections_candidate_nswcs_list)
            sampled_nswcs_list = list( np.random.choice(np.array(candidate_nswcs_list), num_samples, replace=False, p=candidate_nswcs_probs) )

            if len(sampled_nswcs_list)!=num_samples:
                raise ValueError(f"The number of sampled nswcs does not match the number of samples that should be drawn.")

            for sampled_nswcs in sampled_nswcs_list:
                protein_id_to_scrambled_nswcs_list_map[protein_id].append(sampled_nswcs)

                # Increase the number of times the nswcs was picked (i.e. assigned) by 1
                if sampled_nswcs not in nswcs_to_num_picks_map:
                    nswcs_to_num_picks_map[sampled_nswcs] = 1
                else:
                    nswcs_to_num_picks_map[sampled_nswcs] += 1

                # If the number of picks matches or exceeds the real number of connections of a nswcs,
                # the nswcs assignment is exhausted and thus should be removed (from future assignments).
                if nswcs_to_num_connections_map[sampled_nswcs]==1:
                    if 0<nswcs_to_num_picks_map[sampled_nswcs]:
                        removed_nswcs_set.add(sampled_nswcs)
                else:
                    if nswcs_to_num_connections_map[sampled_nswcs]+3<=nswcs_to_num_picks_map[sampled_nswcs]:
                        removed_nswcs_set.add(sampled_nswcs)

        ##################################################################################
        # Step 3: Create map from original to scrambled connections
        ##################################################################################
        # Create the scrambled connection map that maps original connections
        # [dictionary-keys] to scrambled connections [dictionary-values]
        scrambled_connections_map = dict()
        for protein_id in sorted_protein_id_list:
            nswcs_list = protein_id_to_nswcs_list_map[protein_id]
            scrambled_nswcs_list = protein_id_to_scrambled_nswcs_list_map[protein_id]
            if len(nswcs_list)!=len(scrambled_nswcs_list):
                err_msg = f"The number of elements in the original and scrambled nswcs lists are not the same for protein {protein_id}"
                raise ValueError(err_msg)

            for nswcs, scrambled_nswcs in zip(nswcs_list, scrambled_nswcs_list):
                # Scramble a connection with probability p_{scrambling} (i.e. mapping
                # a connection [dictionary-key] to the corresponding scrambled connection 
                # [dictionary-value]).
                u = np.random.uniform(0, 1)
                if u<self.config_dict['train_data_scrambling']['p_scrambling']:
                    scrambled_connections_map[(protein_id, nswcs)] = (protein_id, scrambled_nswcs)

        return scrambled_connections_map

    def get_scrambled_train_set_df(self, 
                                   scrambled_connections_map):
        """
        Use the 'scrambled_connections_map' that maps original connections (i.e. protein-molecule pairs)
        to to be scrambled connections to update the (preprocessed) train (sub)set DataFrame containing
        these scrambled connections.

        Args:
            scrambled_connections_map (dict): Dictionary mapping original connections (i.e. protein-molecule
                pairs) of the form (protein_id, nswcs) to their corresponding scrambled connection
                that will have the same protein_id but a different (i.e. scrambled) nswcs.

        Return:
            (pandas.DataFrame): An updated version of preprocessed train (sub)set with the values in column
                'non_stereochemical_washed_canonical_smiles' 
                moved to the new column 
                'original_non_stereochemical_washed_canonical_smiles' 
                and where the column 
                'non_stereochemical_washed_canonical_smiles' 
                contains now possibly 'scrambled' values.
        
        """
        # Get the (preprocessed) train (sub)set DataFrame
        train_set_df = self.set_to_processed_subset_df_map['train']

        # Loop over all columns of the DataFrame and generate a list of 
        # the 'new' nswcs (i.e. scrambled or identical to orginal)
        # Remark: scrambled_connections_map maps original to new connections
        #         where a connection has the form (protein_id, nswcs).
        orig_connection_list = list(zip(train_set_df['protein_id'], train_set_df['non_stereochemical_washed_canonical_smiles']))
        orig_nswcs_list = list()
        new_nswcs_list  = list()
        for connection in orig_connection_list:
            orig_nswcs = connection[1]
            orig_nswcs_list.append(orig_nswcs)
            # If the connection is a key of the scrambled_connections_map, the original nswcs is
            # replaced by the 'scrambled' nswcs.
            # Remarks: (1) As the protein ID is the same, scrambling a connection means scrambling
            #              the nswcs assignment here.
            #          (2) The dictionary-value scrambled_connections_map[connection] is also a 'connection'
            #              and has the form (protein_id, scrambled_nswcs)
            if connection in scrambled_connections_map:
                new_nswcs = scrambled_connections_map[connection][1]
            else:
                # Otherwise, if the connection is NOT a key of scrambled_connections_map, the connection
                # is not scrambled and the nswcs is thus also NOT scrambled (i.e. the new nswcs 
                # corresponds actually the original nswcs)
                new_nswcs = orig_nswcs
            
            new_nswcs_list.append(new_nswcs)

        # Sanity check, the original nswcs should be the same as the values in the ''non_stereochemical_washed_canonical_smiles'' column
        if list(train_set_df['non_stereochemical_washed_canonical_smiles'])!=orig_nswcs_list:
            err_msg = f"The list of original nswcs does not correspond to the 'non_stereochemical_washed_canonical_smiles' values of the DataFrame."
            raise ValueError(err_msg)

        # Deep-copy the train set DataFrame and create a new column 'original_nswcs'
        # filling it with the current (and actual) nswcs values.
        scrambled_train_set_df = train_set_df.copy(deep=True)
        scrambled_train_set_df['original_non_stereochemical_washed_canonical_smiles'] = scrambled_train_set_df['non_stereochemical_washed_canonical_smiles']

        # Assign the new nswcs to the 'non_stereochemical_washed_canonical_smiles' column
        scrambled_train_set_df['non_stereochemical_washed_canonical_smiles'] = new_nswcs_list

        return scrambled_train_set_df

    def generate_set_to_torch_data_list_map(self):
        """ Generate a map from set (train, valid, test) to the corresponding list of 'torch data objects'."""
        # Information to the user
        self.display_info(f"Generate a map from set (train, valid, test) to the corresponding list of 'torch data objects':")

        # Loop over the (sub)set names and corresponding processed subset DataFrames that are the 
        # key-value pairs of self.set_to_data_fold_indices_map
        set_to_torch_data_list_map = dict()
        for set_name, processed_subset_df in self.set_to_processed_subset_df_map.items():
            # Generate a list of torch data objects (one for each protein-molecule pair) from the processed subset DataFrame
            # Remarks: 1) The method 'get_torch_data_for_pmp' should be applied for each row (=protein-molecule pair) thus axis=1
            self.display_info(f"Generate list of 'torch data objects' for '{set_name}' set...")
            start_time = time.time()
            # Differ cases where processed_subset_df is empty (i.e. there are no datapoints for the current set).
            # Remark: This could happen for the validation set for example.
            if len(processed_subset_df)==0:
                # In case there are no data points, the correpsponding torch_data_list is an empty list
                set_to_torch_data_list_map[set_name] = list()
            else:
                # If there are data points for the current set, generate the corresponding torch data list
                set_to_torch_data_list_map[set_name] = processed_subset_df.apply(lambda x: self.get_torch_data_for_pmp(x), axis=1).tolist()
            
            self.display_info(f"Generation done for {set_name} set. Duration {time.time()-start_time: .2f}s\n")

        # Assign the map to the corresponding class attribute
        self.set_to_torch_data_list_map = set_to_torch_data_list_map


    def get_torch_data_for_pmp(self, 
                               pmp_series):
        """ 
        Return the torch data object constructed for a protein-molecule pair. 
        
        Args:
            pmp_series (pandas.Series): Series object containing the information of a protein-molecule pair.
        
        Return:
            (torch data object): Torch data object of the protein-molecule pair.

        """
        # Get the non-steoreochemical washed canonical SMILES (nswcs) string of the molecule
        nswcs = pmp_series['non_stereochemical_washed_canonical_smiles']

        # Get the molecular graph (dictionary) for the current molecule
        molecule_graph_dict = self.smiles_to_graph_map[nswcs]

        # Get the protein ID
        protein_id = pmp_series['protein_id']

        # Get the protein representation (dictionary) for the protein id
        protein_repr_dict = self.get_protein_representation(protein_id)

        # Define the relation (i.e. 'y') representation (dictionary)
        # Remark: Assign a label 1 if a measurement of the protein-molecule pair is listed on ChEMBL
        #         and otherwise assign label 0.
        relation_repr_dict = {'y': 1 if pmp_series['on_chembl'] else 0}

        # Construct a torch data object
        pmp_torch_data = torch_geometric.data.Data(protein_id=protein_id,
                                                   nswcs=nswcs,
                                                   num_nodes=molecule_graph_dict['num_vertices'],
                                                   **{key: torch.tensor(value) for key, value in molecule_graph_dict.items() if key!='num_vertices'},
                                                   **{key: torch.tensor(value) if not isinstance(value, str) else value for key, value in protein_repr_dict.items()},
                                                   **{key: torch.tensor(value) if not isinstance(value, str) else value for key, value in relation_repr_dict.items()})

        return pmp_torch_data

    def get_protein_representation(self, 
                                   protein_id):
        """
        Return the protein representation (as dictionary) for the current protein.

        Args:
            protein_id (str): Protein ID represented as string (e.g. the protein's ChEMBL ID).

        Return:
            (dict): Protein features dictionary that contains key-value pairs used as attributes
                of the torch.Data object representing a single molecule-protein pair.
        """
        # Check that 'self.protein_representation_params' has been defined (and thus is not None)
        if self.protein_representation_params is None:
            err_msg = f"The attribute 'protein_representation_params' has not been defined, call the method 'determine_global_protein_representation_params' first."
            raise AttributeError(err_msg)

        if self.config_dict['protein_representation_type']=='protein_sequence':
            raise NotImplementedError("Protein sequence featurization has not been implemented.")
        elif self.config_dict['protein_representation_type']=='protein_index':
            protein_repr_dict = {'protein_features': self.protein_representation_params['protein_id_to_protein_index_map'][protein_id]}

        return protein_repr_dict

    def plot_molecular_weight_histogram(self, 
                                        molecular_weights):
        """
        Plot the histogram of the molecular weights.
        """
        # Only make the figures if requested
        if self.make_figs:
            ix = np.where(np.array(molecular_weights)<=1000)
            fig = plt.figure()
            _hist = plt.hist(np.array(molecular_weights)[ix], bins=50, color='b', alpha=0.5, histtype='stepfilled')
            plt.vlines(self.config_dict['molecular_weight_bounds'][0], 0, np.max(_hist[0])*2/3, color='r', lw=2, label='Lower boundary')
            plt.vlines(self.config_dict['molecular_weight_bounds'][1], 0, np.max(_hist[0])*2/3, color='g', lw=2, label='Upper boundary')
            plt.xlabel('Molecular weight [Dalton]')
            plt.xlim([0, 1000])
            plt.ylabel('#Molecules')
            plt.legend()
            fig.tight_layout()

            # Save the figure
            plot_save_path = str( Path(self.figures_dir_path, 'molecular_weight_distribution.png') )
            fig.savefig(plot_save_path)

    def make_number_of_proteins_plot(self, 
                                     num_molecules_dict):
        """ 
        Plot the number of proteins as a function of threshold of molecule number enforced per protein.
        
        Args:
            num_molecules_dict (dict): Dictionary of the form {<protein_id>: <#molecules of the protein>}
        """
        # Only make the figures if requested
        if self.make_figs:
            # The values of the dictionary 'num_molecules_dict' correspond to the number of molecules per protein
            num_molecules_arr = np.array( list(num_molecules_dict.values()) )

            # Define an array containing different thresholds
            num_molecules_thresholds = np.arange(0, num_molecules_arr.max()+2)

            # Loop over an increasing number of molecule number thresholds
            num_proteins = list()
            for num_molecules_threshold in num_molecules_thresholds:
                # Determine the number of proteins that have a number of molecules
                # larger than the current threshold
                num_proteins_threshold = np.sum( num_molecules_threshold<=num_molecules_arr)

                # Append this number to the corresponding list
                num_proteins.append(num_proteins_threshold)

            # Make the number of proteins list an array an create the plot
            num_proteins = np.array(num_proteins)

            # Make the figure
            fig = plt.figure()
            plt.plot(num_molecules_thresholds, num_proteins, 'b-', lw=2)
            plt.vlines(self.config_dict['num_molecules_threshold'], 0, num_proteins.max(), color='r', lw=2, label='Set threshold')
            plt.xlim([0, num_molecules_thresholds.max()+1])
            plt.ylim([0, num_proteins.max()+1])
            plt.xlabel('Molecules threshold')
            plt.ylabel('#Proteins')
            plt.legend()
            fig.tight_layout()

            # Save the figure
            plot_save_path = str( Path(self.figures_dir_path, 'number_of_proteins_plot.png') )
            fig.savefig(plot_save_path)

    
    def plot_num_entries_per_pType(self, 
                                   filtered_df):
        """ Plot the number of entries per pType of entries appearing in the passed filtered DataFrame. """
        # Only make the figures if requested
        if self.make_figs:
            # Count the number of pTypes
            pTypes_counter = collections.OrderedDict( collections.Counter( filtered_df['pType'] ) )

            # Make the figure
            fig = plt.figure()

            # Plot the counts as bars
            x_ticks = range(len(pTypes_counter))
            plt.bar(x_ticks, pTypes_counter.values(), color='b', alpha=0.5)

            # Loop over all pTypes and display the counts also as text
            y_max = 0
            for pType_x, pType_counts in zip(x_ticks, pTypes_counter.values()):
                # Plot the counts as text above the bar
                plt.text(pType_x, pType_counts, pType_counts, horizontalalignment='center', verticalalignment='bottom')
                y_max = max([y_max, pType_counts])

            # Set plot specs
            plt.ylim([0, y_max*1.1])
            plt.xlabel('pType')
            plt.ylabel('#Entries')
            plt.gca().set_xticks(x_ticks)
            plt.gca().set_xticklabels(pTypes_counter.keys())
            fig.tight_layout()

            # Save the figure
            plot_save_path = str( Path(self.figures_dir_path, 'entries_per_type.png') )
            fig.savefig(plot_save_path)

