#combine_chembl_data.py

# Import public modules
import argparse

# Import custom modules
import utils
import chembl_data_combiner

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'combine_chembl_data.py' can only be run as main."
    raise SystemError(err_msg)

# Parse input arguments
parser = argparse.ArgumentParser(description='Combine (and to some extend preprocess) the individual ChEMBL data files of different proteins.')
parser.add_argument('--dev', dest='dev', action='store_true', default=False, help='Flag to be used during development.')
args = parser.parse_args()

# Define a logger object
stream_logging_level = 'DEBUG' if args.dev==True else 'WARNING'
logger = utils.define_logger('./logfiles/combine_chembl_data.log', stream_logging_level=stream_logging_level)

# Define the base directory to the folder where the ChEMBL data was saved for different proteins
ChEMBL_base_dir = './ChEMBL_Data'

# Specify the file paths of the Kinases tables
protein_table_file_path_dict = {
    'kinases': './tables/output/proteins/Kinases_table.tsv',
}

# Define the paths under which the Molecule-Protein-Binding (MPB) datasets of the kinases should be saved in
save_path_dict = {
    'kinases': './tables/output/activities/Activities_Kinases.tsv',
}

# Define all activity comments that will be used to classify the ligand as 'inactive' for the current protein
activity_comments_for_inactives = ['not active', 'inactive', 'inconclusive', 'no inhibition', 'no effect', 'no activity', 'not detected', 'not detectable', 'not significant', 'low activity', 'no significant']

# Define a dictionary that maps standard type categories to be included in the combined data to a list containing the standard_types 
# that might appear in the ChEMBL data for the standrd type category.
standard_type_category_dict = {
    'pic50':    ["pIC50", "Log IC50", "-Log IC50", "pIC50", "log IC50", "Log 1/IC50", "log(1/IC50)", "log1/IC50", "logIC50", "pIC50(app)"],
    'pec50':    ["pEC50", "Log EC50", "log EC50", "-Log EC50", "Log 1/EC50"],
    'pki':      ["pKi", "Log Ki", "pKi", "log Ki", "-Log Ki", "log(1/Ki)", "Log 1/Ki app", "log1/Ki", "logKi", "pKinact"],
    'pk':       ["-ln K", "-Log K", "lnK", "Log 1/K", "log(1/K)", "pK", "pKapp"],
    'pkb':      ["-Log 1/Kb", "-Log K B", "-Log KB", "Log KB", "pKB", "pKb(app)"],
    'pka':      ["-Log KA", "pKa", "pKa app"],
    'pkd':      ["-Log KD", "-Log Kdiss", "Log 1/Kd", "Log Kd", "logKd"],
    'pke':      ["Log Ke"],
    'pkm':      ["Log 1/Km", "log(1/Km)", "pKm"],
    'ic50':     ["IC50"],
    'ec50':     ["EC50"],
    'ki':       ["K inact", "Ki", "Ki app", "Ki app (inact)", "Ki inact", "Ki(app)", "Ki_app"],
    'k':        ["K", "Kapp", "Kact"],
    'kb':       ["K B", "KB", "K Bapp", "K Bind", "KB app", "Kbapp"],
    'ka':       ["Ka"],
    'kd':       ["Kd", "KD app", "KD'", "Kd(app)", "kdiss"],
    'ke':       ["Ke", "Ke(app)"],
    'km':       ["Km", "Km app", "Km'", "Km(app)"],
    'inactive': ["INACTIVE"],
}

# Initialize the combiner object
combiner = chembl_data_combiner.ChEMBLDataCombiner(ChEMBL_base_dir, protein_table_file_path_dict, save_path_dict, logger, standard_type_category_dict, activity_comments_for_inactives)

# Combine the kinases ChEMBL data to one big table each (while performing some preprocessing)
combiner.run() 

# Log some information
logger.info(f"\n\n{'='*100}\nSome remarks concerning the ChEMBL data combination:\n{'='*100}")
logger.info(f"\nRemoved Standard Types:\n{combiner.removed_standard_types}\n")
logger.info(f"\nRemoved Standard Units:\n{combiner.removed_standard_units}\n")
logger.info(f"\nNumber of different combinations of pStandardType entries that have a non-NaN standard unit listed (and thus were removed): (total #{ sum( combiner.non_NaN_standard_units_for_pType_counter.values() ) })\n{combiner.non_NaN_standard_units_for_pType_counter}\n")
logger.info(f"\nThe following {len(combiner.proteins_without_data_dict['kinases'])} Kinases had no data (after preprocessing/filtering): {combiner.proteins_without_data_dict['kinases']}\n")
logger.info("\nThe following molecular ChEMBL IDs correspond to compounds whose non-stereochemical washed canonical SMILES (nswcs) strings are the same and were combined to a single molecule specified by a 'molecule_id':")
for m_id, m_chembl_id_list in combiner._m_id_to_m_chembl_id_list_map.items():
    if 1<len(m_chembl_id_list):
        logger.info(f"{m_id} <- {m_chembl_id_list}")