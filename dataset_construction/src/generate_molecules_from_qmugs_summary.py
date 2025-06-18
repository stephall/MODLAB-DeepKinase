# Import public modules
import argparse
import tqdm
import pandas as pd

# Import custom modules
import utils

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'generate_molecules_from_qmugs_summary.py' can only be run as main."
    raise SystemError(err_msg)

# Parse input arguments
parser = argparse.ArgumentParser(description='Generate a table containing the non-stereochemical washed canonical SMILES (nswcs) strings and molecular weights of all molecules found in the QMugs summary.')
parser.add_argument('--dev', dest='dev', action='store_true', default=False, help='Flag to be used during development.')
args = parser.parse_args()

# Define a logger object
stream_logging_level = 'DEBUG' if args.dev==True else 'WARNING'
logger = utils.define_logger('./logfiles/generate_molecules_from_qmugs_summary.log', stream_logging_level=stream_logging_level)

# Load the QMugs summary table as DataFrame
logger.info("Loading QMugs summary file.")
qmugs_summary_df = pd.read_csv('./tables/input/qmugs_summary.csv')
logger.info(f"Loading done. Number of entries in QMugs summary: {len(qmugs_summary_df)}")

# Get a list of unique smiles strings
unique_smiles = list(set(qmugs_summary_df['smiles']))
logger.info(f"Number of unique SMILES: {len(unique_smiles)}\n")

# Wash these smiles and remove their stereochemistry (if they have one)
# => non-stereochemical washed canonical (nswc) smiles
logger.info(f"Washing all smiles strings (and making them canonical and non-stereochemical): ")
nswc_smiles = [utils.get_washed_canonical_smiles(smiles, remove_stereochemistry=True) for smiles in tqdm.tqdm(unique_smiles)]

# Get the unique non-stereochemical washed canonical SMILES strings and sort them
unique_nswc_smiles = list(set(nswc_smiles))
unique_nswc_smiles.sort()
logger.info(f"Number of unique non-stereochemical washed canonical SMILES strings: {len(unique_nswc_smiles)}")

# Filter the non-measured molecules by their molecular weight 
logger.info("Determine the molecular weight of all these unique molecules")
molecular_weights = [utils.get_molecular_weight(nswcs) for nswcs in tqdm.tqdm(unique_nswc_smiles)]
logger.info("Weight determination done")

# Generate a DataFrame with the non-stereochemical washed canonical SMILES (nswcs) strings and molecular weights of each molecule
# and save it as .tsv file
file_path = './tables/output/molecules/molecules_from_qmugs_summary.tsv'
save_df   = pd.DataFrame({'non_stereochemical_washed_canonical_smiles': unique_nswc_smiles, 'molecular_weight': molecular_weights})
save_df.to_csv(file_path, index=False, sep='\t')

# Log the information
logger.info(f"Saved the table containing the non-stereochemical washed canonical SMILES (nswcs) strings and molecular weight of all QMugs summary molecules in: {file_path}")


                
