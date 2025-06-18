# Import public modules
import argparse
import csv
import os
import tqdm
import pandas as pd
from pathlib import Path

# Import custom modules
import utils

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'generate_nswc_smiles_kinases_chembl.py' can only be run as main."
    raise SystemError(err_msg)

# Parse input arguments
parser = argparse.ArgumentParser(description='Generate a list of the non-stereochemical washed canonical SMILES (nswcs) strings of ALL molecules that have been listed on ChEMBL whose activity has been measured for ANY kinase.')
parser.add_argument('--dev', dest='dev', action='store_true', default=False, help='Flag to be used during development.')
args = parser.parse_args()

# Define a logger object
stream_logging_level = 'DEBUG' if args.dev==True else 'WARNING'
logger = utils.define_logger('./logfiles/generate_nswc_smiles_kinases_chembl.log', stream_logging_level=stream_logging_level)

# Define the base directory containing the ChEMBL data for the Kinases (i.e. ALL measurements for the Kinases we could extract from ChEMBL)
chembl_kinases_base_dir = './ChEMBL_Data/Kinases'

# Check that this is a valid directory
if not os.path.isdir(chembl_kinases_base_dir):
    err_msg = f"Could not find the directory: {chembl_kinases_base_dir}"
    raise FileNotFoundError(err_msg)
    
# Loop over all kinases files
logger.info('Extract all SMILES in the kinases data files')
kinases_chembl_smiles_list = list()
for file_name in tqdm.tqdm(os.listdir(chembl_kinases_base_dir)):
    # If the file has not the correct format, continue to the next iteration
    if not (file_name.startswith('ChEMBL_Data_') and file_name.endswith('.tsv')):
        continue
    
    # Generate the file path
    file_path = Path(chembl_kinases_base_dir, file_name)

    # Open the data as DataFrame
    kinase_chembl_data_df = pd.read_csv(file_path, sep='\t')
    
    # Extract a list of SMILES strings (accessible as 'canonical_smiles') and add them to the global list
    # Remark: Some SMILES entries might be blank (i.e. NaN), drop these from the list before adding it.
    kinases_chembl_smiles_list += list(kinase_chembl_data_df['canonical_smiles'].dropna())

# Log the info
logger.info(f"{len(kinases_chembl_smiles_list)} entries, where the activity of any molecule-kinase pair has been measured, were found on ChEMBL.")

# Get the unique SMILES strings
kinases_chembl_smiles_list = list(set(kinases_chembl_smiles_list))

# Log the info
logger.info(f"{len(kinases_chembl_smiles_list)} molecules, whose activity to any kinase has been measured, were found on ChEMBL.")

# Wash these smiles and remove their stereochemistry (if they have one)
# => non-stereochemical washed canonical (nswc) smiles
logger.info(f"Washing all smiles strings (and making the canonical and non-stereochemical): ")
kinases_chembl_nswc_smiles = [utils.get_washed_canonical_smiles(smiles, remove_stereochemistry=True) for smiles in tqdm.tqdm(kinases_chembl_smiles_list)]

# Get the unique non-stereochemical washed canonical SMILES strings and sort them
kinases_chembl_nswc_smiles = list(set(kinases_chembl_nswc_smiles))
kinases_chembl_nswc_smiles.sort()
logger.info(f"Number of unique non-stereochemical washed canonical SMILES (nswcs) strings: {len(kinases_chembl_nswc_smiles)}")

# Write this list of unique non-stereochemical washed canonical (nswc) SMILES strings to a .tsv file
# Remark: As there will only be one value per row no actual 'tab-separation' is necessary
file_path = './tables/output/molecules/nswc_smiles_kinases_chembl.tsv'
with open(file_path, 'w', newline='') as file:
    # Define the csv writer
    writer = csv.writer(file)
    
    # Write a header row
    writer.writerow(['non_stereochemical_washed_canonical_smiles'])

    # Write each item in a new row
    for smiles in tqdm.tqdm(kinases_chembl_nswc_smiles):
        writer.writerow([smiles])

# Log the information
logger.info(f"Saved the list of these nswcs in the file: {file_path}")