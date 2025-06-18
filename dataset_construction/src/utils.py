# Import public modules
import logging
import os
import rdkit
import tarfile
import pandas as pd
from pathlib import Path

# Import module obtained from 'https://github.com/chembl/ChEMBL_Structure_Pipeline'
import chembl_structure_pipeline

class UniProtTable(object):
    def __init__(self, 
                 uniprot_list_file_path, 
                 logger):
        """
        Args:
            uniprot_list_file_path (str or path): Path to the UniProt list.
            logger (object): Logger object.
        """
        # Assign input to class attribute
        self.uniprot_table_file_path = uniprot_list_file_path
        self.logger                  = logger

        # Load the UniProt table (.tsv file) as pandas.DataFrame
        # Remark: Read all entries as strings (dtype=str)
        self.df = pd.read_csv(self.uniprot_table_file_path, sep='\t', dtype=str)
        
        # Display the columns
        self.logger.info(f"Loaded UniProt table as pandas.DataFrame with {len(self.df)} entries and the columns:\n{list(self.df.columns)}")
        
        # Check if the table only contains entries for humans
        self._check_only_human()
        
    def _check_only_human(self):
        """ Check that the uniprot table only contains entries for the organism 'Homo sapiens (Human)' """
        # Get all unique organisms in the table (/df)
        unique_organisms = list( set(self.df['Organism']) )
        
        # Define the organism
        organism = 'Homo sapiens (Human)'
        
        # Check that organism is contained in unique_organisms
        if not organism in unique_organisms:
            err_msg = f"The organism '{organism}' is not contained in the table!"
            raise ValueError(err_msg)

        if 1<len(unique_organisms):
            err_msg = f"There were multiple organisms and not just the organism '{organism}'.\n The organisms were: {unique_organisms}"
            raise ValueError(err_msg)

        self.logger.info(f"The only organism contained in the table is '{organism}'.")
        
    def extract_quantites(self, 
                          uniprot_id, 
                          quantities):
        """
        Extract quantities from the UniProt table for an entry defined by its uniprot id.
        
        Args:
            uniprot_id (str): Uniprot id of the entry the quantities should be extracted from.
            quantities (list of str): List containing the to be extracted quantities as strings.
            
        Return:
            (dict): Dictionary containing the quantities as key-value pairs.
        
        """
        # Filter the UniProt table by the passed uniprot id
        # Remark: The column 'Entry' contains the uniprot ids of the entries.
        filtered_df = self.df[self.df['Entry']==uniprot_id]
        
        # Check that one unqiue entry was found
        if len(filtered_df)==0:
            err_msg = f"No entries were found for the UniProt ID '{uniprot_id}' in the UniProt table.\nRemark: The column 'Entry' holds the UniProt IDs."
            raise ValueError(err_msg)
        else:
            # In case an entry was found, check that there was only 1
            if 1<len(filtered_df):
                err_msg = f"Multiple entries were found for the UniProt ID '{uniprot_id}' in the UniProt table.\nRemark: The column 'Entry' holds the UniProt IDs."
                raise ValueError(err_msg)
                
        # Loop over the quantities
        quantity_dict = dict()
        for quantity in quantities:
            # Check that the quantity corresponds to a column name of the UniProt table
            if quantity not in self.df.columns:
                err_msg = f"The quantity '{quantity}' is not a column of the UniProt table.\nThe column names are:\n{list(self.df.colnames)}"
                raise ValueError(err_msg)
                
            # Assign the quantity to the dictionary
            # Remark: There is only one entry in filtered_df[quantity], which can be accessed using 'pandas.DataFrame.iloc[0]'
            quantity_dict[quantity] = filtered_df[quantity].iloc[0]
            
        return quantity_dict


def filter_out_uniprot_entry(uniprot_id, 
                             quantities_dict, 
                             filter_quantities_on_dict, 
                             filtered_out_entries_dict):
    """
    Define a function to filter out uniprot entries. 
    
    Args:
        uniprot_id (str): UniProt ID of the UniProt table entry.
        quantities_dict (dict): Dictionary containing the quantities of a UniProt entry.
        filter_quantities_on_dict (dict): Filter dictionary that contains values based on which the 
            quantities are filtered out.
        filtered_out_entries_dict (dict): Default-Dictionary of lists to which UniProt IDs are appended
            in case that they are filtered.

    Return:
        (bool): Boolean flag if the current entry should be filtered out.

    Remark: The dictionary inputs are passed as reference and thus modified by this function.

    """
    # Check that the length of the amino acid sequence matches the value reported for the entry.
    # If it doesn't, filter out the current entry.
    if int(quantities_dict['Length'])!=len(quantities_dict['Sequence']):
        # Append the current uniprot id to the filtered out entries dictionary for key 'Length'
        filtered_out_entries_dict['Length'].append(uniprot_id)

        # The current entry should be filtered out
        return True
        
    # Drop the 'Length' key-value pair
    quantities_dict.pop('Length')
    
    # Check that the 'Protein existence' matches the filtered on value
    # If it doesn't, filter out the current entry.
    if quantities_dict['Protein existence']!=filter_quantities_on_dict['Protein existence']:
        # Append the current uniprot id to the filtered out entries dictionary for key 'Protein existence'
        filtered_out_entries_dict['Protein existence'].append(uniprot_id)

        # The current entry should be filtered out
        return True
        
    # Check that the annotation score is 5
    if float(quantities_dict['Annotation'])!=filter_quantities_on_dict['Annotation']:
        # Append the current uniprot id to the filtered out entries dictionary for key 'Annotation'
        filtered_out_entries_dict['Annotation'].append(uniprot_id)

        # The current entry should be filtered out
        return True

    # If the entry has not been filtered out so far, it should not be filtered out
    return False


def append_quantities_to_output_dict(output_dict, 
                                     uniprot_id, 
                                     quantities_dict):
    """
    Append the quantities and the UniProt ID to their corresponding values in the output dictionary.
    
    Args:
        output_dict (dict): Default dictionary containing lists as values.
        uniprot_id (str): UniProt ID.
        quantities_dict (dict): Dictionary containing the quantities of a UniProt entry.

    Return:
        None

    Remark: The dictionary inputs are passed as reference and thus modified by this function.
    
    """
    # Append the UniProt ID to the corresponding value in the output dictionary
    output_dict['UniProt ID'].append(uniprot_id)
        
    # Loop over the quantities
    for key, value in quantities_dict.items():
        # Rename the key to '<key> (UniProt)'
        key = f"{key} (UniProt)"
    
        # Append the quantity to its corresponding value in the output dictionary
        output_dict[key].append(value)

def get_washed_canonical_smiles(smiles, 
                                remove_stereochemistry=False):
    """
    'Wash' the input SMILES string and return its canonical version.
    
    Args:
        smiles (str): (Canonical) SMILES string.
        remove_stereochemistry (bool): Should we also remove stereochemistry?
            (Default: False)

    Return:
        (str): Washed (canonical) SMILES string.
    
    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Standardize (neutralize) the molecular object
    st_mol = chembl_structure_pipeline.standardize_mol(mol)

    # Remove salt and solvent
    st_mol, _ = chembl_structure_pipeline.get_parent_mol(st_mol)

    # If multiple fragments remain, take the one with the most heavy atoms
    st_mol_frags = rdkit.Chem.GetMolFrags(st_mol, asMols=True, sanitizeFrags=False)
    if 1 < len(st_mol_frags):
        st_mol_frags = sorted(
            st_mol_frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True
        )
        st_mol = st_mol_frags[0]
        
    # If we should remove the stereochemistry from the molecule, remove it
    if remove_stereochemistry:
        rdkit.Chem.RemoveStereochemistry(st_mol) 

    # Get the canonical SMILES string of the 'washed' molecular object and return it
    smiles = rdkit.Chem.MolToSmiles(st_mol, canonical=True)
    return rdkit.Chem.CanonSmiles(smiles)

def get_molecular_weight(smiles):
    """
    Return the molecular weight of the molecule generated by the input SMILES string.
    
    Args:
        smiles (str): (Canonical) SMILES string.

    Return:
        (str): Full molecular weight of the molecule generated by the input SMILES string.
    
    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Determine and return the molecular weight of the RDKit molecule object
    return rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol)


def unpack_tarred_files(base_dir, 
                        logger, 
                        recursion_level=1):
    """
    Recursively unpack tarred files.

    Args:
        base_dir (str or path): Path to the base directory containing the tarred files.
        logger (object): Logger object.
        recursion_level (int): Recursion level indicator.
        
    """
    print(f"Unpacking tarred files in {base_dir}")
    # Unpack all .tar.gz files in the base directory and in their subdirectories
    base_dir_files = os.listdir(base_dir)
    for file_name in base_dir_files:
        # In case the file name starts with '.', continue to next filename
        if file_name.startswith('.'):
            continue
        
        # Deal with the case that the file name ends with '.tar.gz'
        if file_name.endswith('.tar.gz'):
            # Get the 'unpacked file name' (without this extension)
            unpacked_file_name = file_name[:-7]

            # In case the file hasn't been unpacked so far, unpack it.
            if unpacked_file_name not in base_dir_files:
                # In case the file hasn't been unpacked so far, unpack it.
                # Open the file
                with tarfile.open(Path(base_dir, file_name)) as file:
                    # Unpack/extract the file into the current directory
                    file.extractall(base_dir)
                
                # In case that the recursion level is 1, unpack also all the files in this 
                # unpacked file by recursively calling the function
                if recursion_level==1:
                    sub_base_dir = Path(base_dir, unpacked_file_name)
                    unpack_tarred_files(sub_base_dir, recursion_level=2)
                else:
                    # Otherwise continue to next file name in the base dir
                    continue
                    
    logger.info("Unpacking of tarred files in {base_dir} finished.")

def define_logger(log_file_path, 
                  file_logging_level='INFO', 
                  stream_logging_level='DEBUG'):
    """
    Define a logger object and return it.
    
    Args:
        log_file_path (str or path): Path in which the log file should be stored in.
        file_logging_level (str): Logging level for the logs that are stored to the logfile (file).
        stream_logging_level (str): Logging level for the logs that are displayed (stream).

    Return:
        (logging.logger): Logger object.
    
    """
    # Check that the logging levels are expected
    expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if file_logging_level not in expected_levels:
        err_msg = f"The passed (uppercase) file logging level '{file_logging_level.upper()}' is not one of the expected logging level: {expected_levels}"
        raise ValueError(err_msg)
    if stream_logging_level not in expected_levels:
        err_msg = f"The passed (upercase) stream logging level '{stream_logging_level.upper()}' is not one of the expected logging level: {expected_levels}"
        raise ValueError(err_msg)

    # Remove the logfile if there already exists one
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)

    # Turn the file and stream logging levels from strings to actual logging level objects
    file_logging_level   = getattr(logging, file_logging_level.upper())
    stream_logging_level = getattr(logging, stream_logging_level.upper())

    # Get the log file name and use it as name for the logger
    log_file_name = os.path.split(log_file_path)[1]
    logger_name   = log_file_name.removesuffix('.log')

    # Set the root logger's logging level to DEBUG
    # Remark: 1) For each logging event, the root logger's logging level (global) is used to determine if the
    #            event should be logged or not. 
    #         2) Thus, this 'global' logging level oversteers in some sence the 'local' logging levels of the handlers defined below.
    #            As the handler levels are explicitly set below, this should not happen so use the lowest level (DEBUG),
    logging.basicConfig(level=logging.DEBUG)

    # Initialize the logger
    logger = logging.getLogger(logger_name)

    # Remove handlers if the logger has any
    logger.handlers = []

    # Generate a file handler that will store logging information to the log file
    # and add it to the logger
    f_handler = logging.FileHandler(log_file_path)
    f_format  = logging.Formatter(fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_handler.setFormatter(f_format)
    f_handler.setLevel(file_logging_level)
    logger.addHandler(f_handler)

    # Generate a stream handler that will show the logging info to the user
    # and add it to the logger
    s_handler = logging.StreamHandler()
    s_format  = logging.Formatter(fmt='[%(levelname)s]: %(message)s')
    s_handler.setFormatter(s_format)
    s_handler.setLevel(stream_logging_level)
    logger.addHandler(s_handler)

    # Do not propagate logs from the file handlers to the base logger 
    # (so that only the handlers log but not the base logger)
    logger.propagate = False

    return logger