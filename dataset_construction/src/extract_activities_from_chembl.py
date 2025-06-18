#extract_activities_from_chembl.py

# Remark: Parts of the code inspired by https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html

# Import public modules
import argparse
import collections
import datetime
import logging
import os
import time
import tqdm
import pandas as pd
from pathlib import Path

# Import custom modules
import utils

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'extract_activities_from_chembl.py' can only be run as main."
    raise SystemError(err_msg)

# Parse input arguments
parser = argparse.ArgumentParser(description='Extract activities from ChEMBL.')
parser.add_argument('--dev', dest='dev', action='store_true', default=False, help='Flag to be used during development.')
args = parser.parse_args()

# Import the ChEMBL webresource client module
print('Load ChEMBL webresource client module...')
import chembl_webresource_client
from chembl_webresource_client.new_client import new_client
print('Loading done.')
print()

# Initialize ChEMBL accession APIs
target_api   = new_client.target
activity_api = new_client.activity

# Unfortunately, some imported (public) modules use the root logger to log their events
# and because we have to set the root logger level to DEBUG for our events to be logged
# as INFO or DEBUG, this means that these aforementioned modules will start logging their
# INFO or DEBUG events. Circumvent this by explicitly setting their logging level to WARN.
# Remark: This is not an elegant solution but a Hack. The problem is that we should not set
#         the root loggers level from WARN to DEBUG in the first place, but need to do this here.
logging.getLogger('chembl_webresource_client').setLevel(logging.WARN)
logging.getLogger('requests_cache').setLevel(logging.WARN)
logging.getLogger('urllib3').setLevel(logging.WARN)

# Define a logger object
stream_logging_level = 'DEBUG'
logger = utils.define_logger('./logfiles/extract_activities_from_chembl.log', stream_logging_level=stream_logging_level)

# Specify the file paths of the Kinases tables
protein_table_file_path_dict = {
    'kinases': './tables/output/proteins/Kinases_table.tsv',
}

# Specify on what the activities query should be filtered on
# Remark: The 'target_chembl_id' will be updated later by the protein ChEMBL ID for each queried protein 
#         and is therefore initialized to None here.
#activity_query_filters_dict = {'target_chembl_id': None, 'assay_type': "B"}
activity_query_filters_dict = {'target_chembl_id': None}

# Define the base directory for the saved files and generate the folder if it doesn't exist yet
save_base_dir = './ChEMBL_Data'
if not os.path.isdir(save_base_dir):
    os.mkdir(save_base_dir)

# Loop over the protein families (i.e. 'kinases' here)
for protein_family, protein_table_file_path in protein_table_file_path_dict.items():
    logger.info(f"\n\n{'='*100}\nProtein family: {protein_family}\n{'='*100}")
    
    # Load the protein table as pandas.DataFrame
    protein_table_df = pd.read_csv(protein_table_file_path, sep='\t')

    # Initialize an empty list that will be filled with all proteins that are not found on ChEMBL.
    protein_not_found_on_chembl = list()

    # Initialize an empty list that will be filled with all proteins for which no activity data is available on ChEMBL.
    protein_without_entries_on_chembl = list()
    
    # Loop over the proteins (=rows of the table/DataFrame)
    # Remark: The method 'iterrows' returns for each iteration a 2-tuple containing
    #         the row index and the row content as pandas.Series object.
    print(f"Extracting the activities from ChEMBL for all '{protein_family}'")
    for protein_index, protein_series in tqdm.tqdm(protein_table_df.iterrows(), total=len(protein_table_df)):
        # Get the UniProt ID of the current protein
        uniprot_id = protein_series['UniProt ID']

        # The name of the directory that contains the protein files for the current protein family corresponds 
        # to the name of the protein table (of the current family) without the suffix '_table.tsv'
        protein_table_file_name = os.path.split(protein_table_file_path)[1]
        protein_family_dir      = protein_table_file_name.removesuffix('_table.tsv')
        protein_family_path     = str( Path(save_base_dir, protein_family_dir) )
        
        # Generate the directory if it doesn't exist
        if not os.path.isdir(protein_family_path):
            os.makedirs(protein_family_path)
        
        # Define the file name under which the ChEMBL data of the current protein should be saved
        protein_activities_file_name = f"ChEMBL_Data_{uniprot_id}.tsv"
        
        # Construct the file path under which the ChEMBL data of the current protein should be saved in
        protein_activities_file_path = str( Path(protein_family_path, protein_activities_file_name) )
        
        # In case the file already exists, log this, and continue to the next protein
        if os.path.isfile(protein_activities_file_path):
            logger.info(f"Data for protein with UniProt ID '{uniprot_id}' has already been downloaded and saved in: {protein_activities_file_path}")
            continue
        
        ##############################################################################################################
        # 1) Get the protein's ChEMBL ID and access the 'Targets' category on ChEMBL for the UniProt ID for this
        ##############################################################################################################
        # Fetch the target information (corresponding to 'Targets' category on ChEMBL) queried for the UniProt ID of
        # the protein from ChEMBL and restrict the query to specified quantities.
        start_time = time.time()
        targets_query = target_api.get(target_components__accession=uniprot_id).only(
            "target_chembl_id", "organism", "pref_name", "target_type"
        )
        for item in dir(targets_query):
            print(item)

        raise ValueError("STOPPING HERE")
        
        # targets_query is of type 'chembl_webresource_client.query_set.QuerySet', cast it to a pandas.DataFrame
        try:
            targets_df = pd.DataFrame.from_records(targets_query)

            # Remove duplicate rows
            targets_df = targets_df.drop_duplicates()
        except chembl_webresource_client.http_errors.HttpApplicationError:
            

            # Inform the user
            logger.info(f"\nCHEMBL WEBRESOURCE CLIENT HTTP APPLICATION ERROR THROWN FOR PROTEIN WITH UNIPROT ID: {uniprot_id}\n")
            #raise chembl_webresource_client.http_errors.HttpApplicationError

            # Make an empty target DataFrame for the current protein
            targets_df = pd.DataFrame()

        # In case that the targets DataFrame is empty (e.g. no entries on ChEMBL), continue to next protein
        if len(targets_df)==0:
            # Add the protein to the list of proteins that are not found on ChEMBL
            protein_not_found_on_chembl.append(uniprot_id)

            # Log this
            logger.info(f"No 'target information' for the Query of the protein with UniProt ID '{uniprot_id}' => Continue to next protein.")
            continue
        
        # Filter the DataFrame so that only proteins with (lower case) organism 'homo sapiens' and lower case target_type 'single protein'
        targets_df = targets_df[(targets_df['organism'].str.lower()=='homo sapiens') & (targets_df['target_type'].str.lower()=='single protein')]
                
        # Differ cases depending if there is no entry left or there are multiple entries left
        # Remark: Ideally, there should be exactly one entry left, which is then used further down.
        if 0==len(targets_df):
            # Add the protein to the list of proteins that are not found on ChEMBL
            protein_not_found_on_chembl.append(uniprot_id)

            # If there is no entry left, log this, and continue to next protein
            logger.info(f"After removing duplicate rows and filtering by 'organism' (='homo sapiens') and 'target_type' (='single protein') for UniProt ID='{uniprot_id}' Query, no entries remain => Continue to next protein.")
            continue
        if 1<len(targets_df):
            # In case there are multiple entries left, throw an error
            err_msg = f"After removing duplicate rows and filtering by 'organism' (='homo sapiens') and 'target_type' (='single protein') for UniProt ID='{uniprot_id}' Query, multiple target entries remain:\n{targets_df}"
            raise ValueError(err_msg)

        # Get the protein's ChEMBL ID
        p_chembl_id = targets_df['target_chembl_id'].iloc[0]
        
        ##############################################################################################################
        # 2) Extract the activity data
        ##############################################################################################################
        # Fetch the activity by filtering on the target IDs on the protein's ChEMBL ID and an other quantities specified
        # in 'activity_query_filters_dict'. Moreover, restrict the query to specified quantities
        # Assign the protein ChEMBL ID as value to the key 'target_chembl_id' in the activity query filter dictionary
        activity_query_filters_dict['target_chembl_id'] = p_chembl_id

        # Query the activities for the current protein
        protein_activities_query = activity_api.filter(
            **activity_query_filters_dict
        ).only(
            "activity_id",
            "assay_chembl_id",
            "assay_description",
            "assay_type",
            "standard_value",
            "standard_units",
            "standard_type",
            "activity_comment",
            "molecule_chembl_id",
            "canonical_smiles",
            "relation",
            "year",
            "data_validity_comment",
            "target_chembl_id",
            "target_organism",
            "type",
        )

        # protein_activities_query is of type 'chembl_webresource_client.query_set.QuerySet', cast it to a pandas.DataFrame
        logger.info(f"Casting the query object to a pandas.DataFrame for protein with UniProt ID '{uniprot_id}' (start time {datetime.datetime.now()})...")
        start_time = time.time()
        protein_activities_df = pd.DataFrame.from_records(protein_activities_query)
        logger.info(f"Casting of query object to pandas.DataFrame for protein with UniProt ID '{uniprot_id}' finished. Duration: {(time.time()-start_time)/60:.2f}min")

        # In case that the DataFrame has no entries, inform the user and continue to next protein
        if 0==len(protein_activities_df):
            # Add the protein to the list of proteins for which there is no data available on ChEMBL
            protein_without_entries_on_chembl.append(uniprot_id)

            # Log this
            logger.info(f"No activity entries for the protein with UniProt ID '{uniprot_id}' => Continue to next protein.")

            # In case that the stream handler has logging level 'WARNING' this information would not be shown to the user, 
            # thus inform the user about it using print.
            logger_handler_names = [str(handler_obj) for handler_obj in logger.handlers]
            continue

        # Check that the DataFrame only contains the filtered values for the filtered quantities and warn the user if it does.
        for col_name, filtered_on_value in activity_query_filters_dict.items():
            set_diff = set(protein_activities_df[col_name])-set([filtered_on_value])
            if 0<len(set_diff):
                logger.warning(f"The column '{col_name}' was filtered based on the entry '{filtered_on_value}' for protein with UniProt ID '{uniprot_id}' but contains the following (not expected) values: {list(set_diff)}")

        # Save the DataFrame as .tsv file
        protein_activities_df.to_csv(protein_activities_file_path, sep='\t', index=False)

        # Log that the data file has been saved
        logger.info(f"Saved the downloaded data for protein with UniProt ID '{uniprot_id}' in the file: {protein_activities_file_path}")

    # Log information concerning which proteins where not found on ChEMBL
    logger.info(f"\nRemark 1: The following {len(protein_not_found_on_chembl)} '{protein_family}' were NOT found on ChEMBL: {protein_not_found_on_chembl}")

    # Log information concerning which proteins where found but for which no activity data was available
    logger.info(f"\nRemark 2: The following {len(protein_without_entries_on_chembl)} '{protein_family}' were found on ChEMBL but no activity data was available for them: {protein_without_entries_on_chembl}")

# Log that extraction is finished
logger.info(f"\n\n{'='*100}\nExtraction finished.\n{'='*100}")
