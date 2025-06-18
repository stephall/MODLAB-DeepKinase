#generate_kinases_table.py

# Import public modules
import argparse
import collections
import time
import tqdm
import pandas as pd

# Import custom modules
import utils

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'generate_kinases_table.py' can only be run as main."
    raise SystemError(err_msg)

# Parse input arguments
parser = argparse.ArgumentParser(description='Generate a table for all (Human) Kinases.')
parser.add_argument('--dev', dest='dev', action='store_true', default=False, help='Flag to be used during development.')
args = parser.parse_args()

# Define a logger object
stream_logging_level = 'DEBUG' if args.dev==True else 'WARNING'
logger = utils.define_logger('./logfiles/generate_kinases_table.log', stream_logging_level=stream_logging_level)

# Define the file path to the input table containing all Kinases retrieved from KinHub: http://www.kinhub.org/kinases.html
# Supposed to be from article: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1433-7
# Downloaded: 25.08.2022
KinHubList_filepath = './tables/input/KinHub_List.xlsx'

# Define the file path to the input table containing all Kinases retrieved from UniProt for the keyword 'Kinases' ('KW-0418').
# This list has been obtained using the keyword 'Kinases' on UniProt (i.e. searched for 'Kinase' on UniProt and then select ID 'KW-0418') 
# filtered by Status 'Reviewed' and Orginism 'Human'.
# Downloaded: 10.02.2023
KeywordKinases_filepath = './tables/input/Keyword-Kinases(KW-0418)_UniProt.tsv'

# Define the file path to the UniProt (Human) List
uniprot_table_filepath = './tables/input/UniProt_Human_list.tsv'

# Define the path for the output table containing the UniProt Information of all Kinases
output_filepath = './tables/output/proteins/Kinases_table.tsv'

# Define the to be extracted quantities
quantities = ['Entry Name', 'Protein names','Protein existence', 'Annotation', 'Organism', 'Reviewed', 'Length', 'Sequence']

# Define a dictionary with the values of the quantities the entries should be filtered on
filter_quantities_on_dict = {
    # This will filter entries with 'Protein existence' of 'Evidence at protein level' and thus not include entries with 'Protein existence' 
    # of for example 'Evidence at transcript level', 'Uncertain', 'Predicted', or 'Inferred from homology'.
    'Protein existence': 'Evidence at protein level',
    
    # Only keep entries with a maximal annotation score of 5
    'Annotation': 5.0,
}

# Load the Excel (.xlsx) file containing the table KinHub listed kinases
KinHubList_df = pd.read_excel(KinHubList_filepath)

# Load the keyword kinases (.tsv) file containing the Keyword Kinases (UniProt) kinases 
KeywordKinases_df = pd.read_csv(KeywordKinases_filepath, sep='\t')

# Display which kinases are found in which source [kinases found on KinHub with the ones found 
# on UniProt for Keyword 'Kinase' ('reviewed' and 'Human')], their intersection, and their union.
KinHubList_uniprot_ids     = list( set(KinHubList_df['UniprotID']) )
KeywordKinases_uniprot_ids = list( set(KeywordKinases_df['Entry']) )
intersection_uniprot_ids   = list( set(KinHubList_uniprot_ids).intersection(set(KeywordKinases_uniprot_ids)) )
union_uniprot_ids          = list( set(KinHubList_uniprot_ids).union(set(KeywordKinases_uniprot_ids)) )

# Log the info
logger.info(f"{len(KinHubList_uniprot_ids)} kinases are found on KinHub.")
logger.info(f"{len(KeywordKinases_uniprot_ids)} kinases are found on UniProt for the keyword 'Kinase' ('reviewed' and 'Human').")
logger.info(f"{len(intersection_uniprot_ids)} kinases were found in both of these sources (intersection).")
logger.info(f"{len(union_uniprot_ids)} kinases were found in total (union).\n")

# Construct a list of all kinases UniProt IDs found on both sources combined, that first lists all kinases only found on
# KinHub and then afterwards lists all the kinases not found on KinHub but on UniProt with keyword 'Kinase'.
all_kinases_uniprot_ids = KinHubList_uniprot_ids + list( set(KeywordKinases_uniprot_ids)-set(KinHubList_uniprot_ids) )

# (Sanity) check that the resulting list has the same element as the union defined above
if set(all_kinases_uniprot_ids)!=set(union_uniprot_ids):
    err_msg = f"The set-union of the kinases found in both sources is does not have the same elements as the list of all kinases (KinHub+set-difference)."
    raise ValueError(err_msg)

# (Sanity) check that the resulting list has unique elements
if len(set(all_kinases_uniprot_ids))!=len(all_kinases_uniprot_ids):
    err_msg = f"The list of all kinases (KinHub+set-difference) contains certain UniProt IDs more than once."
    raise ValueError(err_msg)

# Load the .tsv file containing a big table of UniProt obtained using an advanced search with Taxonomy "Homo sapiens (Human/Man) [9606]" and then selecting "Human (204,961)" 
# in the side bar of Popular organisms so that it is indeed only "Homo sapiens (Human)" (and not e.g. neaderthalensis etc.)
# Downloaded: 25.08.2022
uniprot_table = utils.UniProtTable(uniprot_table_filepath, logger)

# Initialize the dictionary that will contain the filtered out entries (i.e. their UniProt IDs) as default dictionary (containing lists).
filtered_out_entries_dict = collections.defaultdict(list)

# Intialize the output table as default dictionary (containing lists).
output_dict = collections.defaultdict(list)

# Initialize a list that will be filled with entries not found in the UniProt table (i.e. their UniProt IDs)
not_found = list()

# Loop over the UniProt IDs in the input table
print(f"Generating the Kinases table from {uniprot_table_filepath} using the Kinases listed in the files {KinHubList_filepath} and {KeywordKinases_filepath}.")
start_time = time.time()
for uniprot_id in tqdm.tqdm(all_kinases_uniprot_ids):
    # Try to extract the quantities for the current uniprot id
    try:
        # Extract the quantities for the current uniprot id
        quantities_dict = uniprot_table.extract_quantites(uniprot_id, quantities)
        
        # Check if the current entry should be filtered out
        filter_out_flag = utils.filter_out_uniprot_entry(uniprot_id, quantities_dict, filter_quantities_on_dict, filtered_out_entries_dict)
        
        # If it should be filtered out, continue to next iteration/entry
        if filter_out_flag==True:
            continue
        
        # Append the uniprot_id and the quantities to their corresponding output_dict values
        utils.append_quantities_to_output_dict(output_dict, uniprot_id, quantities_dict)

        # Append boolean flag to the output dictionary that indicates if the kinase is listed on KinHub or not
        kinase_listed_on_KinHub = (uniprot_id in KinHubList_uniprot_ids)
        output_dict['kinase_listed_on_KinHub'].append(kinase_listed_on_KinHub)
                        
    except ValueError:
        not_found.append(uniprot_id)

# Log that generation of the table worked
logger.info(f"Generation of Kinases table from {uniprot_table_filepath} using the Kinases listed in the files {KinHubList_filepath} and {KeywordKinases_filepath} finished. Duration: {time.time()-start_time:.2f}s")

# Log which proteins were not found in the UniProt Table
logger.info(f"The following {len(not_found)} Kinases were not found in UniProt Human Table: {not_found}")

# Log why certain protein entries were removed/not included in the data
for key, value in filtered_out_entries_dict.items():
    if key=='Length':
        logger.info(f"Removed {len(value)} protein entries because 'Length' was not equal to the length of 'Sequence': {value}")
    else:
        logger.info(f"Removed {len(value)} protein entries because '{key}' was not equal to '{filter_quantities_on_dict[key]}': {value}")

# Transform the output dictionary to a pandas.DataFrame
output_df = pd.DataFrame.from_dict(output_dict)

# Save this dataframe as .tsv file and log where it was saved
output_df.to_csv(output_filepath, sep='\t', index=False)
logger.info(f"Saved generated Kinase table (containing {len(output_df)} Kinases) as '{output_filepath}'")