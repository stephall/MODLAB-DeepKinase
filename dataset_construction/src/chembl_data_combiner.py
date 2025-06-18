#chembl_data_combiner.py

# Import public modules
import collections
import functools
import joblib
import math
import multiprocessing
import os
import time
import tqdm
import pandas as pd
from pathlib import Path

# Import custom modules
import utils

class ChEMBLDataCombiner(object):
    def __init__(self, 
                 ChEMBL_base_dir, 
                 protein_table_file_path_dict, 
                 save_path_dict, 
                 logger, 
                 standard_type_category_dict, 
                 activity_comments_for_inactives, 
                 dev=False):
        """
        Args:
            ChEMBL_base_dir (str or path): Path to base directory containing the extracted ChEMBL files.
            protein_table_file_path_dict (dict): Dictionary containing the path to the protein tables.
            save_path_dict (dict): Dictionary containing the paths in which files should be saved in.
            logger (object): Logger object.
            standard_type_category_dict (dict): Dictionary containing various standard type categories
                as dictionary-keys and a list containing various alternative names for this category as 
                dictionary-values.
            activity_comments_for_inactives (list): List of activity comments used to denote inactivity.
            dev (bool): Boolen flag to indicate if the data combiner should be run in development mode
                (dev=True) or not (dev=False).
                (Default: dev=False).
        """
        # Assign inputs to class attributes
        self.ChEMBL_base_dir                  = ChEMBL_base_dir
        self.protein_table_file_path_dict     = protein_table_file_path_dict
        self._save_path_dict                  = save_path_dict
        self._logger                          = logger
        self._standard_type_category_dict     = standard_type_category_dict
        self._activity_comments_for_inactives = activity_comments_for_inactives
        
        # Generate a class attribute dictionary that maps the standard types encountered on ChEMBL 
        # (values of 'self._standard_type_category_dict') to its associated (standard type) category 
        # (keys of 'self._standard_type_category_dict').
        # Remark: Make the standard types all lower case.
        self._standard_type_to_category_map = dict()
        for standard_type_category, standard_type_list in self._standard_type_category_dict.items():
            for standard_type in standard_type_list:
                self._standard_type_to_category_map[standard_type.lower()] = standard_type_category

        # Generate a list of to be filtered on standard types from the values of 'self._standard_type_category_dict'
        # Remark: Make all the standard types lower case.
        filtered_on_standard_types       = functools.reduce(lambda x, y: x+y, self._standard_type_category_dict.values(), [])
        self._filtered_on_standard_types = [standard_type.lower() for standard_type in filtered_on_standard_types]

        # Initialize 'self._removed_standard_types', which will be filled with all standard types that are not in
        # filtered_on_standard_types and thus removed from the data set, as empty list.
        self._removed_standard_types = list()

        # Initialize 'self._removed_standard_units', which will be filled with all standard units of entries that are removed
        # when constructing the 'pX=-log10(X)' column values, as empty list.
        self._removed_standard_units = list()

        # Initialize 'self.proteins_without_data_dict' as empty defaults dictionary (lists) that will hold the UniProt
        # IDs of the proteins that have no data (either from the start or after preprocessing/filtering)
        self.proteins_without_data_dict = collections.defaultdict(list)

        # Initialize an empty list that will hold all the molecular ChEMBL IDs and canonical SMILES strings tuples
        self._m_chembl_id_canonical_smiles_tuples = list()

        # Initialize an empty dictionary that will map the molecular chembl ID to their corresponding non-stereochemical
        # washed canonical SMILES (nswcs) string (and also the inverse)
        self._m_id_to_nswcs_map = dict()
        self._nswcs_to_m_id_map = dict()

        # Initialize an empty dictionary that will map the molecular ChEMBL ID to the molecule ID.
        # The reason that this mapping and the distrinction between molecule ID and molecular ChEMBL ID has to 
        # be constructed is that there might be multiple molecules (with different ChEMBL IDs listed on ChEMBL) that have 
        # the same non-stereochemical washed canonical SMILES string. As they have the same biological effect inside the 
        # body, they should not be treated separately but should be combined, which is done by mapping them to 
        # the same molecule ID.
        self._m_chembl_id_to_m_id_map = dict()

        # Initialize an empty dictionary that will map the molecule ID to all the molecular ChEMBL IDs that correspond to it.
        self._m_id_to_m_chembl_id_list_map = dict()

        # Initialize an empty dictionary that will map the molecular ChEMBL ID to the full molecular weight
        # of the molecule generated by the non-stereochemical washed canonical SMILES string.
        self._m_id_to_molecular_weight_map = dict()

        # Initialize a dictionary that will contain all the column names of the ChEMBL data for both protein families
        self._chembl_data_col_names_dict = dict()

        # Initialize a list that will contain tuples of standard_type_category and standard_unit for all non-NaN 
        # standard units with a standard_type_categories starting with 'p'.
        self._non_NaN_standard_units_for_pType = list()

    @property
    def removed_standard_types(self):
        """ Remove the unique removed standard types (obtained from self._removed_standard_types). """
        return list( set(self._removed_standard_types) )
    
    @property
    def removed_standard_units(self):
        """ Remove the unique removed standard units (obtained from self._removed_standard_units). """
        return list( set(self._removed_standard_units) )

    @property
    def non_NaN_standard_units_for_pType_counter(self):
        """
        Return a Counter object containing tuples the non-NaN standard units 
        and their standard type categories (starting with 'p')
        """
        return collections.Counter(self._non_NaN_standard_units_for_pType)

    def run(self, 
            n_jobs=None):
        """
        Combine the ChEMBL data of the kinases (protein families) 
        into two big tables (one for each of the protein families).

        Remarks: 1) Perform some preprocessing; filtering and adding of new columns.
                 2) Preprocessing, filtering, and combining the protein data is rather quick,
                    but washing the molecular SMILES strings is rather slow. Use multi-processing to
                    wash these SMILES string (=> specificy the number of jobs as input argument).

        Args:
            n_jobs (int or None): How many jobs to use to wash the molecular SMILES strings.
                    If None, the computers CPU number will be used [multiprocessing.cpu_count()].
                    (Default None)

        """
        # Initialize an empty dictionary for the dataset DataFrames of the two protein families (protein_family: dataset_df)
        dataset_df_dict = dict()

        # Loop over the protein families (only family is 'kinases' here)
        for protein_family, protein_table_file_path in self.protein_table_file_path_dict.items():
            # Display which protein family we currently are dealing with
            self._logger.info(f"\n\n{'='*100}\nProtein family: {protein_family}\n{'='*100}")
            
            # Load the protein table as pandas.DataFrame
            protein_table_df = pd.read_csv(protein_table_file_path, sep='\t')

            # Initialize an empty list that will hold the preprocessed DataFrames of all proteins
            df_list = list()
            
            # Loop over the proteins (=rows of the table/DataFrame)
            # Remark: The method 'iterrows' returns for each iteration a 2-tuple containing
            #         the row index and the row content as pandas.Series object.
            print(f"Combining the ChEMBL entries for all '{protein_family}'")
            for _, protein_series in tqdm.tqdm( protein_table_df.iterrows(), total=len(protein_table_df)):
                # Get the UniProt ID of the current protein
                uniprot_id = protein_series['UniProt ID']

                # The name of the directory that contains the protein files for the current protein family corresponds 
                # to the name of the protein table (of the current family) without the suffix '_table.tsv'
                protein_table_file_name = os.path.split(protein_table_file_path)[1]
                protein_family_dir      = protein_table_file_name.removesuffix('_table.tsv')
                protein_family_path     = str( Path(self.ChEMBL_base_dir, protein_family_dir) )
                
                # Generate the directory if it doesn't exist
                if not os.path.isdir(protein_family_path):
                    os.makedirs(protein_family_path)
                
                # Define the file name under which the ChEMBL data of the current protein should be saved
                protein_activities_file_name = f"ChEMBL_Data_{uniprot_id}.tsv"
                
                # Construct the file path under which the ChEMBL data of the current protein should be saved in
                protein_activities_file_path = str( Path(protein_family_path, protein_activities_file_name) )
                
                # In case the file does not exist, append the protein's UniProt ID to the list of proteins without data 
                # for the current protein family, and continue to next protein
                if not os.path.isfile(protein_activities_file_path):
                    self._logger.info(f"No Protein data available for the protein with UniProt ID '{uniprot_id}'.")
                    self.proteins_without_data_dict[protein_family].append(uniprot_id)
                    continue
                
                # Load the file as pandas.DataFrame
                protein_activities_df = pd.read_csv(protein_activities_file_path, sep='\t')

                # Set the dtype of some columns in the DataFrame
                protein_activities_df = protein_activities_df.astype({'activity_comment': str})

                # Check the ChEMBL data column names
                self._check_chembl_data_col_names(protein_family, protein_activities_df.columns.to_list())

                # Remove duplicate rows in the DataFrame
                protein_activities_df = protein_activities_df.drop_duplicates()
                
                # Keep only entries that do not have a data validity comment (so the corresponding entry is empty <=> parsed as NaN)
                # Remark: The method to search for NaN is 'isnull' and not 'isnan' for pandas.Series objects
                filtered_df = protein_activities_df[protein_activities_df['data_validity_comment'].isnull()]

                # Keep only entries that do not have a lower case activity comment of 'not determined'
                filtered_df = filtered_df[filtered_df['activity_comment'].str.lower()!='not determined']

                # Keep only entries that have a defined canonical SMILES string
                filtered_df = filtered_df[~filtered_df['canonical_smiles'].isnull()]
                            
                # Filter on the all expected standard types and collect the ones that are excluded
                standard_types_before_filtering = set(filtered_df['standard_type'].str.lower())
                filtered_df                     = filtered_df[filtered_df['standard_type'].str.lower().isin(self._filtered_on_standard_types)]
                standard_types_after_filtering  = set(filtered_df['standard_type'].str.lower())
                set_diff                        = standard_types_before_filtering-standard_types_after_filtering
                self._removed_standard_types   += list(set_diff)
                
                # In case that the filtered DataFrame has no entries, append the protein's UniProt ID to the list of proteins 
                # without data for the current protein family, and continue to next protein
                if 0==len(filtered_df):
                    self._logger.info(f"No entries left for protein with UniProt ID '{uniprot_id}' after filtering, continue to next protein")
                    self.proteins_without_data_dict[protein_family].append(uniprot_id)
                    continue
                
                # Determine the 'standard type category' for each entry
                standard_type_category = filtered_df.apply(lambda x: self._standard_type_to_category_map[x['standard_type'].lower()], axis='columns')
                filtered_df['standard_type_category'] = standard_type_category
                
                # Determine 'pX' and 'pType' for each entry and concatenate them as new columns to the DataFrame
                # Remark: As two variables are returned by the method 'self._determine_pX_and_pType', we need to pass the
                #         key-word argument result_type='expand'.
                applied_df         = filtered_df.apply(lambda x: self._determine_pX_and_pType(x['standard_value'], x['standard_type_category'], x['standard_units']), axis='columns', result_type='expand')
                applied_df.columns = ['pX', 'pType']
                filtered_df        = pd.concat([filtered_df, applied_df], axis='columns')
                
                # Determine the 'effect_comment' for each entry (based on the activity comment and pX value listed on ChEMBL) 
                # and add it as column to the DataFrame
                filtered_df['effect_comment'] = filtered_df.apply(lambda x: self._determine_effect_comment(x['activity_comment'], x['pX']), axis='columns')

                # Remove all entries that have no activity comment (it is None => null in Pandas) and have a NaN pX value (=> null in pandas)
                num_before_filtering = len(filtered_df)
                filtered_df = filtered_df[~( filtered_df['effect_comment'].isnull() & filtered_df['pX'].isnull() )]
                self._logger.info(f"Removed {num_before_filtering-len(filtered_df)} entries for protein with UniProt ID '{uniprot_id}' because their pX value was None/NaN while they did not have any effect comment.")

                # In case that the filtered DataFrame has no entries, append the protein's UniProt ID to the list of proteins 
                # without data for the current protein family, and continue to next protein
                if 0==len(filtered_df):
                    self._logger.info(f"No entries left for protein with UniProt ID '{uniprot_id}' after filtering (remove 'acitive' and pX=NaN), continue to next protein")
                    self.proteins_without_data_dict[protein_family].append(uniprot_id)
                    continue

                # Remove duplicate rows in the DataFrame
                # Remark: As duplicate rows have already been removed in the unfiltered (non-preprocessed) DataFrame
                #         there shouldn't be any duplicate rows in the filtered DataFrame, but still do it to make sure.
                filtered_df = filtered_df.drop_duplicates()

                # Add the protein information (contained in protein_series) to each row.
                # First, generate a pandas.DataFrame whose rows all contain the information of the protein series object 
                # (i.e. the series keys become columns with values that are all equivalend) and which has the same number 
                # of rows as the filtered DataFrame.
                # Remark: To achiev this, create a list containing N=#rows (of the filtered DataFrame) copies of the 
                #         protein series object, concatenate these series along the columns and transpose them to
                #         obtain a DataFrame with the series keys as columns.
                protein_info_df = pd.concat([protein_series for _ in range( len(filtered_df) )], axis='columns').T

                # Second, concatenate this protein information DataFrame to the filtered DataFrame
                # Remark: The indices of the filtered and protein infor DataFrames must be reset because pandas.concat attempts
                #         to combine rows with the same index causing the insertion of NaN values if indices don't match.
                #         As all rows of protein_info_df are equivalent, index matching isn't necessary and even problematic,
                #         so reset the index, which will lead to the expected concatenation.
                filtered_df = pd.concat([filtered_df.reset_index(drop=True), protein_info_df.reset_index(drop=True)], axis='columns')

                # Obtain a list of (molecular) ChEMBL IDs and canonical SMILES strings and add it to the corresponding global list
                self._m_chembl_id_canonical_smiles_tuples += list( zip(filtered_df['molecule_chembl_id'], filtered_df['canonical_smiles']) )

                # Append the filtered DataFrame to the list of DataFrames
                df_list.append(filtered_df)

            # Concatenate the DataFrames in the list to obtain the dataset DataFrame for the current protein family
            dataset_df_dict[protein_family] = pd.concat(df_list)
        
        # Construct the molecular ChEMBL ID maps
        self._construct_m_chembl_id_maps(n_jobs=n_jobs)

        # Loop over the dataset_df_dict
        for protein_family, dataset_df in dataset_df_dict.items():
            # Add a new column containing the molecular IDs of the dataset that correspond to the lowest ChEMBL IDs of the set of ChEMBL IDs 
            # for a non-stereochemical washed canonical SMILES string (see PROBLEM comment in method '_construct_m_chembl_id_maps').
            dataset_df['molecule_id'] = dataset_df.apply(lambda x: self._m_chembl_id_to_m_id_map[x['molecule_chembl_id']], axis='columns')

            # Add a new column containing the non-stereochemical washed canonical SMILES string of the molecule for an entry given its molecule ID
            dataset_df['non_stereochemical_washed_canonical_smiles'] = dataset_df.apply(lambda x: self._m_id_to_nswcs_map[x['molecule_id']], axis='columns')

            # Add a new column containing the molecular weight of the molecule (generated by the non-stereochemical washed canonical SMILES string)
            dataset_df['molecular_weight'] = dataset_df.apply(lambda x: self._m_id_to_molecular_weight_map[x['molecule_id']], axis='columns')

            # Change the name of all the columns appearing in the original ChEMBL data to '<column_name> (ChEMBL)'
            # Remark: One exception is the column 'target_chembl_id' which is changed to 'protein_id' without any postfix
            chembl_data_col_names_map = {col_name: f"{col_name} (ChEMBL)" for col_name in self._chembl_data_col_names_dict[protein_family]}
            chembl_data_col_names_map['target_chembl_id'] = 'protein_id'
            dataset_df = dataset_df.rename(columns=chembl_data_col_names_map)

            # Reorder the columns
            col_names_order = self._get_col_names_order(dataset_df.columns.to_list())
            dataset_df      = dataset_df.reindex(columns=col_names_order)

            # Store the dataset as .tsv file
            dataset_df.to_csv(self._save_path_dict[protein_family], sep='\t', index=False)

    def _check_chembl_data_col_names(self, 
                                     protein_family, 
                                     col_names):
        """ 
        Check the ChEMBL data column names are the same for all files for one protein family
        and throw an error otherwise.

        The check is done against the values of 'self._chembl_data_col_names_dict' with the protein family as key.
        In case the key doesn't exist, add the passed col_names as values to the key.
        
        Args:
            protein_family (str): Name of the protein family (i.e. 'kinases' is expected here).
            col_names (list): Column names that should be checked.
        
        """
        # If protein_family is not a key of 'self._chembl_data_col_names_dict', add it and the col_names
        # as key-value pair and return
        if protein_family not in self._chembl_data_col_names_dict:
            self._chembl_data_col_names_dict[protein_family] = col_names
            return

        # Otherwise, check that the value of 'self._chembl_data_col_names_dict' to the key protein_family
        # contains the same items as col_names (checked by making both sets) and throw an error if it doesn't.
        if set(self._chembl_data_col_names_dict[protein_family])!=set(col_names):
            err_msg = f"The passed column names\n{col_names}\ncontains names that were not column names in all the ChEMBL "\
                      f"data files so far, which were:\n{self._chembl_data_col_names_dict[protein_family]}"
            raise ValueError(err_msg)

    def _construct_m_chembl_id_maps(self, 
                                    n_jobs=None):
        """
        Construct the molecular ChEMBL ID maps. 
        
        Args:
            n_jobs (int or None): How many jobs to use to wash the molecular SMILES strings.
                If None, the computers CPU number will be used [multiprocessing.cpu_count()].
                (Default None)
        
        """
        # Parse the number of jobs to use to wash the molecular SMILES strings
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()

        # Get the unique set of molecular ChEMBL IDs and canonical SMILES strings combinations
        # and extract the corresponding molecular ChEMBL IDs and canonical SMILES strings.
        m_chembl_id_canonical_smiles_tuples = list( set(self._m_chembl_id_canonical_smiles_tuples) )
        m_chembl_id_list                    = [item[0] for item in m_chembl_id_canonical_smiles_tuples]
        canonical_smiles_list               = [item[1] for item in m_chembl_id_canonical_smiles_tuples]

        # Wash all canonical SMILES strings and remove stereochemistry
        # Rem: This will create non-stereochemical washed canonical SMILES (nswcs) strings
        print(f"\nWashing all canonical SMILES strings, while also removing any stereochemistry (using {n_jobs} parallel jobs)...")
        start_time = time.time()
        if n_jobs==1:
            # Do non-parallel execution for 1 job
            nswcs_list = tqdm.tqdm( map(lambda x: utils.get_washed_canonical_smiles(x, remove_stereochemistry=True), canonical_smiles_list), total=len(canonical_smiles_list), position=0, leave=True)
        else:
            # Otherwise, use parallel execution using 'joblib'
            nswcs_list = joblib.Parallel(n_jobs=4)(joblib.delayed(utils.get_washed_canonical_smiles)(x, remove_stereochemistry=True) for x in tqdm.tqdm(canonical_smiles_list))
        self._logger.info(f"Washed all canonical SMILES strings from ChEMBL entries and removed their stereochemistry (using {n_jobs} parallel jobs). Duration: {(time.time()-start_time)/60:.2f}min")

        # Find the unique combinations of molecular ChEMBL IDs and non-stereochemical washed canonical SMILES (nswcs) strings
        m_chembl_id_nswcs_tuples = list( set( zip(m_chembl_id_list, nswcs_list) ) )
        m_chembl_id_list         = [item[0] for item in m_chembl_id_nswcs_tuples]
        nswcs_list               = [item[1] for item in m_chembl_id_nswcs_tuples]
        
        # Check that the mapping from molecular ChEMBL ID to non-stereochemical washed canonical SMILES (nswcs) string is unique
        if len( set(m_chembl_id_list) )<len(m_chembl_id_nswcs_tuples):
            err_msg = f"Some of the molecular ChEMBL IDs correspond to multiple non-stereochemical washed canonical SMILES strings."
            raise ValueError(err_msg)

        # Problem:
        # It can happen that different molecules on ChEMBL have the same non-stereochemical washed canonical SMILES string because the 
        # molecules become the same when washed and made non-stereochemical (Example 'CHEMBL1785021' and 'CHEMBL2070048').
        # Solution:
        # These molecules should be combined to one with a unique 'molecule_id' as their properties inside the body depend on the 
        # non-stereochemical washed molecular structure. 
        # Use the lowest ChEMBL ID as 'molecule_id' (e.g. the rows with molecular ChEMBL IDs 'CHEMBL1785021' and 'CHEMBL2070048' will 
        # get a the same molecule ID corresponding to 'CHEMBL1785021').

        # First, generate a mapping from the non-stereochemical washed canonical SMILES string to their ChEMBL ID.
        # Example: This will for example generate the key-value pair:
        #          'CCOc1ccc2cc(-c3nn(C(C)C)c4ncnc(N)c34)ccc2c1': ['CHEMBL1785021', 'CHEMBL2070048']
        nswcs_to_m_chembl_id_list_dict = collections.defaultdict(list)
        for nswcs, m_chembl_id in zip(nswcs_list, m_chembl_id_list):
            # Append the molecular ChEMBL ID to the value of the dictionary
            nswcs_to_m_chembl_id_list_dict[nswcs].append(m_chembl_id)
            
            # Sort the value (list of molecular ChEMBL IDs) and reassign it
            nswcs_to_m_chembl_id_list_dict[nswcs] = self._sort_chembl_id_list( nswcs_to_m_chembl_id_list_dict[nswcs] )
            
        
        # Get a unique list of non-stereochemical washed canonical SMILES (nswcs) string, sort it and generate a map from the
        # this nswcs to the molecule ID that corresponds to 'MID_<index>'. Also create the inverse map.
        unique_nswcs_list       = list(set(nswcs_list))
        self._nswcs_to_m_id_map = {nswcs: f"MID_{nswcs_index}" for nswcs_index, nswcs in enumerate(unique_nswcs_list)}
        self._m_id_to_nswcs_map = {m_id: nswcs for nswcs, m_id in self._nswcs_to_m_id_map.items()}
        
        # Loop over the molecular ChEMBL ID lists corresponding to the different non-stereochemical washed canonical SMILES strings and 
        # generate a map from the molecular ChEMBL ID to the molecule ID.
        # Example: This will for example generate the key-value pairs:
        #          'molecular_chembl_id': 'molecule_id'
        #          'CHEMBL1785021':       'CHEMBL1785021'
        #          'CHEMBL2070048':       'CHEMBL1785021'
        # Also generate the 'inverse' map from molecule IDs to a list of the different molecular ChEMBL IDs it corresponds to
        # Example: This will for example generate the key-value pairs:
        #          'molecule_id':   'molecular_chembl_id_list'
        #          'CHEMBL1785021': ['CHEMBL1785021', CHEMBL2070048']
        # Remark: The m_chembl_id_lists are sorted by ChEMBL ID in increasing order, so the first corresponds to the lowest ID
        #         which is declared as 'molecule_id'.
        m_chembl_id_to_m_id_map      = dict()
        m_id_to_m_chembl_id_list_map = dict()
        for nswcs, m_chembl_id_list in nswcs_to_m_chembl_id_list_dict.items():
            # Get the molecule ID
            m_id = self._nswcs_to_m_id_map[nswcs]
            
            # 1) Add the molecule ID as value to all the molecular ChEMBL IDs to 'm_chembl_id_to_m_id_map'
            for m_chembl_id in m_chembl_id_list:
                m_chembl_id_to_m_id_map[m_chembl_id] = m_id

            # 2) Add the molecular ChEMBL ID list as value to the molecule ID to 'm_id_to_m_chembl_id_list_map'
            m_id_to_m_chembl_id_list_map[m_id] = m_chembl_id_list

        # Check that the keys of 'self._m_id_to_nswcs_map' correspond to the unique set of values of 'm_chembl_id_to_m_id_map'
        if set(self._m_id_to_nswcs_map.keys())!=set(m_chembl_id_to_m_id_map.values()):
            err_msg = f"Some of keys of 'self._m_id_to_nswcs_map' are not values of 'm_chembl_id_to_m_id_map'."
            raise ValueError(err_msg)

        # Assign the dictionaries to their corresponding class attributes
        self._m_chembl_id_to_m_id_map      = m_chembl_id_to_m_id_map
        self._m_id_to_m_chembl_id_list_map = m_id_to_m_chembl_id_list_map

        # Generate a dictionary mapping the molecular ChEMBL ID to the molecular weight of the molecule 
        # (generated from the non-stereochemical washed canonical SMILES string)
        print("\nDetermine all molecular weights...")
        start_time = time.time()
        self._m_id_to_molecular_weight_map = {m_chembl_id: utils.get_molecular_weight(smiles) for m_chembl_id, smiles in tqdm.tqdm( self._m_id_to_nswcs_map.items() )}
        self._logger.info(f"Determined all molecular weights. Duration: {time.time()-start_time:.2f}s")

    def _get_col_names_order(self, 
                             col_names):
        """
        Determine and return the order of the columns for the to be saved table. 
        
        Arg:
            col_names (list): List of all column names.

        Return:
            (list): List with the new order for all the column names.
        
        """
        # Initialize the column names dictionary as empty defaults dictionary that will hold
        # different column categories as keys and their column names in a list as values.
        col_names_dict = collections.defaultdict(list)

        # Define a priority order list for some of the custom column names
        col_names_dict['custom_priority'] = ['protein_id', 'molecule_id', 'non_stereochemical_washed_canonical_smiles', 'pX', 'pType', 'effect_comment', 'molecular_weight']
        
        for col_name in col_names:
            # Differ cases depending to which category the current column name belongs to
            if col_name.endswith('(ChEMBL)'):
                # Column values from ChEMBL
                col_names_dict['chembl'].append(col_name)
            elif col_name.endswith('(UniProt)'):
                # Column values from UniProt
                col_names_dict['uniprot'].append(col_name)
            elif col_name.endswith('(KinHub_List)') or col_name.endswith('(NHRList)'):
                # Column values from one of the protein family (i.e. simply 'kinases' here) specific lists
                col_names_dict['protein_family'].append(col_name)
            else:
                # Otherwise, it must be a custom column name so check if the current
                # name has some priority and has therefore already been defined in the
                # value (list) of col_names_dict['custom_priority'].
                # If it is not included, add it to the rest of the custom column names
                if col_name not in col_names_dict['custom_priority']:
                    col_names_dict['custom_rest'].append(col_name)

        # Construct and return the new column names order
        return col_names_dict['custom_priority'] + col_names_dict['custom_rest'] + col_names_dict['chembl'] + col_names_dict['uniprot'] + col_names_dict['protein_family']

    def _determine_pX_and_pType(self, 
                                standard_value, 
                                standard_type_category, 
                                standard_units):
        """
        Determine the pX=-log10(X) of an entry depending on the standard type category and the standard units.
        
        Args:
            standard_value (float or NaN): The standard value of an entry.
            standard_type_category (str): The standard type category of an entry.
            standard_units (str): The standard units of an entry.

        Returns:
            (float or None, str): The pX and pType values of the entry.

        Remarks: 1) As these entries will be added to a new column in a DataFrame, None values will become NaN.
                 2) Outside this method (inside method 'run'), all entries with pX (that are not innactive) will
                    be filtered out (/removed).
        """
        # Differ cases for the standard type category
        if standard_type_category=='inactive':
            # If the standard type category is 'inactive', ensure that the value is NaN
            if not math.isnan(standard_value):
                err_msg = f"The standard type category is 'inactive' but the standard value is not NaN, but {standard_value}."
                raise ValueError(err_msg)
            
            # Return None for pX, and 'inactive' for pType
            return None, 'inactive'

        elif standard_type_category.startswith('p'):
            # If the standard type category starts with 'p' the standard_value corresponds to pX.

            # Check that the standard units are neither a string nor NaN
            if isinstance(standard_units, str) or not math.isnan(standard_units):
                # Append a tuple of the standard type category and the standard units to the list that keeps
                # track of these non-NaN standard unit standard type categories starting with 'p'.
                self._non_NaN_standard_units_for_pType.append( (standard_type_category, standard_units) )

                # Return None for pX and the standard type category for pType
                return None, standard_type_category
            
            # Return the standard value for pX and the standard type category for pType
            return standard_value, standard_type_category
            
        else:
            # Otherwise, assume that the standard value corresponds to X and thus transform it to pX here.

            # Generate the pType by prefixing 'p' to the standard type category
            pType = f"p{standard_type_category}"

            # In case that the standard value is None, return None for pX here together with the pType
            if math.isnan(standard_value):
                return None, pType

            # Transform the standard value depending on the units
            # Remark: Any value X can be written as X=standard_value*10^{units_exp} where exp is the exponent 
            #         associated with the standard units (e.g. units_exp=-9 for 'nM').
            #         Thus pX = -log10(X) = -log10(standard_value)-log10(10^{units_exp}) = -log10(standard_value)-units_exp
            # Determine the exponents associated with the standard units
            if standard_units=='nM':
                units_exp = -9.0
            else:
                # Add the standard units to 'self._removed_standard_units'
                # Remark: None pX values (for all but inactive entries) will be filtered out later (outside this method).
                self._removed_standard_units.append(standard_units)

                # Return None for pX and the pType
                return None, pType
                
            # Calculate pX = -log10(X) = -log10(standard_value)-units_exp and return it
            try:
                pX = -math.log10(standard_value)-units_exp
            except ValueError as e:
                self._logger.error(f"math.log10(standard_value) threw a ValueError. The standard value is {standard_value}, the standard units are {standard_units}, and the standard type category is {standard_type_category}.")
                raise ValueError(e)

            return pX, pType

    def _determine_effect_comment(self, 
                                  activity_comment_chembl, 
                                  pX):
        """ 
        Determine the effect comment for a single entry based on the ChEMBL activity comment
        and the (ChEMBL) pX value of the entry.

        Args:
            activity_comment_chembl (str): Activity comment of an entry.
            pX (NaN or float): The pX value of an entry.

        Return:
            (str or None): The effect comment of the entry.
        """
        # Check if the activity comment (listed on ChEMBL) is one of the activity comments that are used
        # for inactive compounds (defined in class attribute 'self._activity_comments_for_inactives')
        if str(activity_comment_chembl).lower() in self._activity_comments_for_inactives:
            # In case that pX is NaN, use 'labeled_inactive' as effect comment, else use None as effect comment
            if math.isnan(pX):
                return 'ineffective'
            else:
                return None

        # Otherwise, return None as effect comment
        return None

    def _sort_chembl_id_list(self, 
                             chembl_id_list):
        """
        Sort the list of input ChEMBL IDs.
        
        Arg:
            chembl_id_list (list): List containing ChEMBL IDs to be sorted.
        
        Return:
            (list): The sorted list.

        """
        # The ChEMBL IDs have the form "CHEMBL<ID>"
        def chembl_id_numeral(chembl_id_str):
            return int( chembl_id_str.replace('CHEMBL', '') )

        # Sort the chembl_id_list
        chembl_id_list.sort(key=chembl_id_numeral)

        # Return it
        return chembl_id_list

