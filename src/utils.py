# utils.py

# Import public modules
import hashlib
import hydra
import logging
import omegaconf
import os
import re
import rdkit
import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem import Descriptors

# Import module obtained from 'https://github.com/chembl/ChEMBL_Structure_Pipeline'
import chembl_structure_pipeline

# Import custom modules
from . import model_storage
from . import model_handler_factory

def sort_chembl_id_list(chembl_id_list):
    """
    Sort the list of input ChEMBL IDs.
    
    Arg:
        chembl_id_list (list): List containing ChEMBL IDs to be sorted.
    
    Return:
        None

    Remark: This list is passed by reference and can thus be manipulated within 
            this function, so that the sorted list doesn't need to be returned
    """
    # The ChEMBL IDs have the form "CHEMBL<ID>"
    def chembl_id_numeral(chembl_id_str):
        return int( chembl_id_str.replace('CHEMBL', '') )

    # Sort the chembl_id_list
    chembl_id_list.sort(key=chembl_id_numeral) 

def sort_target_id_list(target_id_list):
    """
    Sort the list of input target IDs.
    
    Arg:
        target_id_list (list): List containing target IDs to be sorted.
    
    Return:
        None

    Remark: This list is passed by reference and can thus be manipulated within 
            this function, so that the sorted list doesn't need to be returned
    """
    # The target IDs have the simple form "CHEMBL<ID>" where the "CHEMBL<ID>" (and thus target_id=p_chembl_id) 
    # or the more complex form "CHEMBL<ID>_<extension>" where the "<extension>" is a distinction of the protein
    # (e.g. 'agonist' or 'antagonist').
    def target_id_numeral(target_id_str):
        return int( target_id_str.replace('CHEMBL', '').split('_')[0] )

    # Sort the target IDs
    # Remark: The first sort will sort the extensions (if there is one) for each protein
    target_id_list.sort()                       # In-place
    target_id_list.sort(key=target_id_numeral)  # In-place

def define_logger(log_file_path, 
                  file_logging_level='INFO', 
                  stream_logging_level='DEBUG'):
    """
    Define a logger object and return it.
    
    Args:
        log_file_path (str or Path): Path in which the log file should be stored in.
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

    # Generate the directory for the log files if it does not already exist
    # Remark: 'os.path.split' splits a file path into the path to the directory and the file name
    #         that is returned as tuple (dir_file_path, file_name).
    log_file_dir_path = os.path.split(log_file_path)[0]
    if not os.path.isdir(log_file_dir_path):
        os.makedirs(log_file_dir_path)

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
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_format  = logging.Formatter(fmt='[%(levelname)s]: %(message)s')
    s_handler.setFormatter(s_format)
    s_handler.setLevel(stream_logging_level)
    logger.addHandler(s_handler)

    # Do not propagate logs from the file handlers to the base logger 
    # (so that only the handlers log but not the base logger)
    logger.propagate = False

    return logger

def set_global_cuda_settings(logger=None, 
                             make_cuda_operations_deterministic=False):
    """
    Fix specific 'global' CUDA settings.

    Args:
        logger (logger object or None): Logger to be used (if passed and not None)
            to display information.
            (Default: None)
        make_cuda_operations_deterministic (bool): Boolean flag that specifies if the 
            CUDA operations should be made deterministic.
            (Default: False)

    Return:
        None
    """
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm for the used hardware
    torch.backends.cudnn.benchmark = False
    if logger is not None:
        logger.info(f"Set torch.backends.cudnn.benchmark to {torch.backends.cudnn.benchmark}\n")
    else:
        print(f"Set torch.backends.cudnn.benchmark to {torch.backends.cudnn.benchmark}\n")

    # In case that the CUDA operations should be made deterministic, fix further settings
    if make_cuda_operations_deterministic:
        # Inform the user that the CUDA operations should be made deterministic
        if logger is not None:
            logger.info("Make CUDA operations deterministic:")
        else:
            print("Make CUDA operations deterministic:")
        
        # Certain operations in Cudnn are not deterministic, and the following line will force them to behave
        torch.backends.cudnn.deterministic = True
        if logger is not None:
            logger.info(f" => torch.backends.cudnn.deterministic has been set to {torch.backends.cudnn.deterministic}")
        else:
            print(f" => torch.backends.cudnn.deterministic has been set to {torch.backends.cudnn.deterministic}")

        # Set the CUBLAS Workspace configuration
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
        if logger is not None:
            logger.info(f" => os.environ['CUBLAS_WORKSPACE_CONFIG'] has been set to '{os.environ['CUBLAS_WORKSPACE_CONFIG']}'\n")
        else:
            print(f" => os.environ['CUBLAS_WORKSPACE_CONFIG'] has been set to '{os.environ['CUBLAS_WORKSPACE_CONFIG']}'\n")


def map_tracked_list_to_tracked_df(tracked_list):
    """
    Map the tracked list from a list of tuple in the form [(epoch, values),...] 
    to a pandas.DataFrame of two columns of the form [(epochs), (values)].

    Args:
        tracked_list (list): List of 2-tuples of the form [(epoch, values),...].

    Return:
        (pandas.DataFrame): Dataframe containing the 'epochs' and 'values' as columns.
    """
    # Use unpacking to obtain the epochs and the values as lists
    epochs, values = zip(*tracked_list)

    # Define a dataframe with these lists and return this dataframe
    return pd.DataFrame(data={'epochs': epochs, 'values': values})

def map_tracked_df_to_tracked_list(tracked_df):
    """
    Map the tracked_df from a pandas.DataFrame of two columns of the form 
    DataFram[(epochs), (values)] to a tracked list of the form [(epoch, value), ...].

    Args:
        tracked_df (pandas.DataFrame): Dataframe containing the 'epochs' and 'values' as columns.

    Return:
        (list): List of 2-tuples of the form [(epoch, value),...].
    """
    # Zip the tracked_df's columns, make it a list and return it
    return list( zip(tracked_df['epochs'], tracked_df['values']) )


def plot_learning_curve(save_dir_path, 
                        epoch='last', 
                        axs=None, 
                        plot_specs=dict(), 
                        plot_save_path=None):
    """
    Plot the learning curve up to the current epoch. 

    Args:
        save_dir_path (str or Path): Path to the directory in which the model parameters
            (and also the metrics) have been saved in during training of a model.
        epoch (int or str): Epoch to plot the learning curve for either as positive integer
            or as string 'last'.
            (Default: 'last')
        axs (plt.Axis object): Axis for the plot.
            (Default: None -> Generate a new axis object)
        plot_specs (dict): Plot specifications.
            (Default: dict() -> Use default plot specs.)
        plot_save_path (str or Path): Path in which the plot should be saved in.
            (Default: None -> Don't save the plot.)
        
    """
    # Define the default plot specs in the internal plot specs '_plot_specs'
    _plot_specs = {
        'alpha': 0.5,
        'axis_fs': 20,
        'leg_fs': 15,
        'tick_fs': 15,
        'ms': 7,
        'colors': ['r', 'b', 'g', 'k', 'c', 'y', 'orange', 'purple'],
    }

    # Treat the case where no axis is passed
    axs_passed = True
    if axs is None:
        # Make a figure
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))

        # Set the axis passed boolean flag to False
        axs_passed = False
    
    # Update the internal plot specs with the passed plot specs
    _plot_specs.update(plot_specs)

    # Define a model storage object
    storage = model_storage.ModelStorage(save_dir_path)

    # Get the tracked list dictionary saved for the epoch
    tracked_list_dict = storage.get_tracked_list_dict(epoch)

    # Loop over the tracked list dictionary, while extract metrics and losses.
    metric_df_dict = dict()
    loss_df_dict   = dict()
    for tracked_quantity in tracked_list_dict:
        # Differ cases where the tracked quantity contains 'metric' or 'loss'.
        # Remark: In case it doesn't contain these, the quantity is not ploted later.
        # In both cases, map the tracked lists from a lists of tuple in the form 
        # [(epoch, value),...] to a pandas.DataFrames of two columns with the 'epochs' 
        # and 'values'.
        if 'metric' in tracked_quantity:
            metric_df_dict[tracked_quantity] = map_tracked_list_to_tracked_df(tracked_list_dict[tracked_quantity])
        elif 'loss' in tracked_quantity:
            loss_df_dict[tracked_quantity] = map_tracked_list_to_tracked_df(tracked_list_dict[tracked_quantity])
        else:
            # Do nothing for other quantities
            pass
    
    # Get the x-axis and y-axis limits
    max_epoch  = 0
    max_loss   = 0
    max_metric = 0
    # Loop over the different losses
    for loss_df in loss_df_dict.values():
        # Update the max epoch and max loss values
        max_epoch = max([max_epoch, loss_df['epochs'].max()])
        max_loss  = max([max_loss, loss_df['values'].max()])
    # Loop over the different metrics
    for metric_df in metric_df_dict.values():
        # Update the max epoch and max metric values
        max_epoch  = max([max_epoch, metric_df['epochs'].max()])
        max_metric = max([max_metric, metric_df['values'].max()])
        
    ###############################################################################################################
    # (1) Plot the losses
    ###############################################################################################################
    # Get the axis of the losses
    ax = axs[0]

    # Loop over the losses
    for loss_index, (loss_name, loss_df) in enumerate(loss_df_dict.items()):
        # Get the color
        color = _plot_specs['colors'][loss_index]

        # Plot the loss values
        ax.plot(loss_df['epochs'], loss_df['values'], 'o-', alpha=_plot_specs['alpha'], 
                color=color, label=loss_name, ms=_plot_specs['ms'])

    # Set plot specs for axis
    ax.set_xlim([0, max_epoch+1])
    ax.set_ylim([0, max_loss*1.05])
    ax.set_xlabel('Epoch', fontsize=_plot_specs['axis_fs'])
    ax.set_ylabel('Losses', fontsize=_plot_specs['axis_fs'])
    ax.legend(fontsize=_plot_specs['leg_fs'])
    ax.tick_params(axis='both', labelsize=_plot_specs['tick_fs'])

    ###############################################################################################################
    # (2) Plot the metrics
    ###############################################################################################################
    # Get the axis of the metrics
    ax = axs[1]

    # Loop over the metrics
    for metric_index, (metric_name, metric_df) in enumerate(metric_df_dict.items()):
        # Get the color
        color = _plot_specs['colors'][metric_index]

        # Plot the metrics values
        ax.plot(metric_df['epochs'], metric_df['values'], 'o-', alpha=_plot_specs['alpha'], 
                color=color, label=metric_name, ms=_plot_specs['ms'])

    # Set plot specs for axis
    ax.set_xlim([0, max_epoch+1])
    ax.set_ylim([0, max_metric*1.05])
    ax.set_xlabel('Epoch', fontsize=_plot_specs['axis_fs'])
    ax.set_ylabel('Metrics', fontsize=_plot_specs['axis_fs'])
    ax.legend(fontsize=_plot_specs['leg_fs'])
    ax.tick_params(axis='both', labelsize=_plot_specs['tick_fs'])

    # In case no axis was passed, plot the figure
    if not axs_passed:
        # Make a tight layout
        plt.tight_layout(h_pad=3.0)
        plt.show()

    # Save the figure if a save path was passed (and thus plot_save_path is not None)
    if plot_save_path is not None:
        # Generate the directory if it doesn't exist yet
        # Remark: os.path.split splits a path into (directory_path, file_name)
        plot_dir_path = os.path.split(plot_save_path)[0]
        if not os.path.isdir(plot_dir_path):
            os.makedirs(plot_dir_path)

        # Save the figure in the passed path
        fig.savefig(plot_save_path)


def get_config_dict(config_dir_rel_path='../configs', 
                    config_name='config.yaml', 
                    overrides=[]):
    """
    Use hydra compose to load the config file.
    
    Args:
        config_dir_rel_path (str): Relative path to the directory containing the config files.
            (Default: './configs')
        config_name (str): Name of the to config file to be loaded.
            (Default: 'config.yaml')
        overrides (list): List of stringified conditions (e.g. ["param==new_value", ...]) used
            to override default values similar to the way how it would be done via command line.
            (Default: [])
    
    Return:
        (dict): Config dictionary.
    
    """
    # Determine the (absolute path) to the directory containing the config files
    config_dir_abs_path = os.path.abspath(config_dir_rel_path)

    # Read the configurations as omegaconf DictConfig object using hydra compose
    with hydra.initialize_config_dir(config_dir_abs_path, version_base="1.1"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        
    # Return the cfg (omegaconf.dictconfig.DictConfig object) as Python dictionary
    return map_dictconfig_to_dict(cfg)

def map_dictconfig_to_dict(dictconfig):
    """ Map a 'omegaconf.dictconfig.DictConfig' to a normal python dictionary. """
    # Initialize the empty output dictionary
    output_dict = dict()

    # Loop over the keys and values of the omegaconf DictConfig object
    for key, value in dictconfig.items():
        # Differ cases where the value is itself an omegaconf DictConfig object or not
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            # In case that the value is itself an omegaconf DictConfig object,
            # recursively call the function itself on the value and assign the
            # resulting dictionary as value to the key in the output dictionary.
            output_dict[key] = map_dictconfig_to_dict(value)
        else:
            # Differ cases where the value is an omegaconf.listconfig.ListConfig object or not
            if isinstance(value, omegaconf.listconfig.ListConfig):
                # Transform the omegaconf.listconfig.ListConfig object into a normal list
                output_dict[key] = [item for item in value]
            else:
                # Assign the key-value pair to the output dictionary
                output_dict[key] = value

    # Return the output dictionary
    return output_dict

def construct_output_run_dir_path(passed_args, 
                                  output_base_dir, 
                                  config_dir_rel_path, 
                                  config_name):
    """
    Construct the path to the output file directory of the run for 'hydra' based on the passed arguments.

    Args:
        passed_args (list of str): List containing the arguments passed to the script (as strings).
        output_base_dir (str): The base directory of the ouputs.
        config_dir_rel_path (str): Relative path to the directory containing the config files.
            (Default: './configs')
        config_name (str): Name of the to config file to be loaded.
            (Default: 'config.yaml')

    Return:
        (str): The output file directory path of the run as string.
    
    """
    # In case no arguments were passed, use 'defaults' as directory name
    if len(passed_args)==0:
        run_dir_name = 'defaults'
    else:
        # Otherwise, use the joined passed arguments as postfix
        # Define the name of the output directory for the current run
        run_dir_name = '|'.join(passed_args)

        # Replace '=' by '\=' in the name
        # Remark: Otherwise, hydra can not partse the argument 'hydra.run.dir=<dir_path>'
        #         in case there are multiple (non-escaped) '=' signs.
        run_dir_name = re.sub(r'=', '\=', run_dir_name)

        # Hash the run_dir name
        # First byte encode the run directory name (using 'ascii' encoding)
        byte_run_dir_name = bytes(run_dir_name, 'ascii')

        # Second, hash it using the 'md5' hash algorithm of the hashlib module
        # and return it as hexadecimal number used as run directory name
        hashed_run_dir_name = hashlib.md5( byte_run_dir_name ).hexdigest()

        # Make this hash a string
        run_dir_name = str( hashed_run_dir_name )

    # Construct the output files directory path and return it
    # Remarks: 1) '${now:%Y-%m-%d}' will create a folder with the date the code has been run
    #          2) '${model.name}' will create a folder with the model name
    return str( Path(output_base_dir, 'outputs', '${now:%Y-%m-%d}', '${model.name}', '${label}', run_dir_name) )

def parse_output_base_dir():
    """ Parse the base directory of the output files from the passed argument (sys.argv). """
    # Define a default output base directory that will be overwritten in case
    # that the output base directory was passed as argument
    output_base_dir = './'

    # Loop over the arguments
    output_base_dir_pattern = r'--output_base_dir=(.*)'
    output_base_dir_index = None
    for index, arg in enumerate(sys.argv):
        if re.match(output_base_dir_pattern, arg):
            # Get the output base directory
            output_base_dir = re.findall(output_base_dir_pattern, arg)[0]

            # Store the current index as the one to be removed below
            output_base_dir_index = index

    # In case that the output base directory was found (i.e. output_base_dir_index is 
    # not None anylonger) remove it from the arguments list by its index
    if output_base_dir_index is not None:
        sys.argv.pop(output_base_dir_index)

    return output_base_dir

def load_all_nswcs(output_folder_path):
    """
    Return all non-stereochemical washed canonical SMILES (nswcs) strings 
    that were present in the the train, validation, or test set for the when
    training the model whose training outputs are in 'output_folder_path'.

    Args:
        output_folder_path (str): Path to the output folder of a trained model.
            This is used to construct the processed dataset and extract all nswcs used for model
            training and evaluation.

    Return:
        (list): List of all nswcs strings that are present in the the train, validation, or test set.
    """
    # Define the model handler
    model_handler, config_dict = model_handler_factory.define_model_handler(output_folder_path, load_data=True)
    
    # Load the model
    model_handler.load()
    
    # Extract all non-stereochemical washed canonical SMILES (nswcs) strings for the train, validation, and test sets
    set_name_to_nswcs_map = dict()
    for set_name, subset_df in model_handler.data_handler.data_preprocessor.set_to_processed_subset_df_map.items():
        set_name_to_nswcs_map[set_name] = list(set(subset_df['non_stereochemical_washed_canonical_smiles']))
    
    # Sanity check: There should not be an overlap of nswcs between any pair of the three sets
    set_names = list(set_name_to_nswcs_map.keys())
    for set_index_1, set_name_1 in enumerate(set_names):
        for set_index_2, set_name_2 in enumerate(set_names):
            if set_index_1<set_index_2:
                nswcs_1 = set(set_name_to_nswcs_map[set_name_1])
                nswcs_2 = set(set_name_to_nswcs_map[set_name_2])
                nswcs_intersection = nswcs_1.intersection(nswcs_2)
                num_intersections = len(nswcs_intersection)
                print(f"Number of intersecting molecules ({set_name_1}/{set_name_2}): {num_intersections}")
    
    # Create a list of all nswcs (in all sets)
    all_nswcs = list()
    for nswcs in set_name_to_nswcs_map.values():
        all_nswcs += nswcs
    
    print(f"Number of unique molecules: {len(all_nswcs)}")
    
    return all_nswcs

#####################################################################################################################################
### Chem-informatics utils
#####################################################################################################################################
def get_molecular_weight_from_smiles(smiles):
    """
    Return the molecular weight of a molecule specified by its passed SMILES string. 

    Args:
        smiles (str): SMILES string of a molecule.

    Return:
        (rdkit.Chem.rdchem.Mol): Molecular object.
    
    """
    # Generate the molecular object
    mol_obj = rdkit.Chem.MolFromSmiles(smiles)

    return Descriptors.ExactMolWt(mol_obj)

def get_nswc_smiles(smiles):
    """
    Return the non-stereochemical washed canonical SMILES (nswcs) string of a molecule.
    Remark: This function combines the functionality of 'get_washed_canonical_smiles' and 'get_nsc_smiles',
            while only requiring to generate a molecular object once.
    
    Args:
        smiles (str): SMILES string of a molecule.

    Return:
        (str): The non-stereochemical washed canonical SMILES (nswcs) string of the molecule.
    
    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol_obj = rdkit.Chem.MolFromSmiles(smiles)

    # Standardize (neutralize) the molecular object
    st_mol_obj = chembl_structure_pipeline.standardize_mol(mol_obj)

    # Remove salt and solvent
    st_mol_obj, _ = chembl_structure_pipeline.get_parent_mol(st_mol_obj)

    # If multiple fragments remain, take the one with the most heavy atoms
    st_mol_frags = rdkit.Chem.GetMolFrags(st_mol_obj, asMols=True, sanitizeFrags=False)
    if 1 < len(st_mol_frags):
        st_mol_frags = sorted(
            st_mol_frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True
        )
        st_mol_obj = st_mol_frags[0]

    # Remove the Steoreochemistry from this object (this updates 'st_mol_obj' in-place)
    rdkit.Chem.RemoveStereochemistry(st_mol_obj) 

    # Get the canonical SMILES string from the molecular object and return it
    # Remark: 'canonical=True' should make the SMILES string canonical in 'rdkit.Chem.MolToSmiles', 
    #         but we still want to ensure this by using 'rdkit.Chem.CanonSmiles()' on the results.
    nswc_smiles = rdkit.Chem.MolToSmiles(st_mol_obj, canonical=True)
    return rdkit.Chem.CanonSmiles(nswc_smiles)

def get_nsc_smiles(smiles):
    """
    Remove stereochemistry from a molecular SMILES string and return this non-stereochemical canonical (nsc) SMILES string.
    
    Args:
        smiles (str): SMILES string of a molecule.

    Return:
        (str): Non-stereochemical canonical SMILES string of the molecule.
    
    """
    # Get the molecular object of the input molecule
    molecule_obj = rdkit.Chem.MolFromSmiles(smiles)

    # Remove the Steoreochemistry from this object (this updates 'molecule_obj' in-place)
    rdkit.Chem.RemoveStereochemistry(molecule_obj) 

    # Get the canonical SMILES string from the molecular object and return it
    # Remark: 'canonical=True' should make the SMILES string canonical in 'rdkit.Chem.MolToSmiles', 
    #         but we still want to ensure this by using 'rdkit.Chem.CanonSmiles()' on the results.
    nsc_smiles = rdkit.Chem.MolToSmiles(molecule_obj, canonical=True)
    return rdkit.Chem.CanonSmiles(nsc_smiles)

def get_washed_canonical_smiles(smiles):
    """
    Wash the input SMILES string and return its canonical version.
    
    Args:
        smiles (str): SMILES string of a molecule.

    Return:
        (str): Washed canonical SMILES string of the molecule.
    
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

    # Get the canonical SMILES string of the 'washed' molecular object and return it
    smiles = rdkit.Chem.MolToSmiles(st_mol, canonical=True)
    return rdkit.Chem.CanonSmiles(smiles)

def draw_molecule(mol):
    """
    Draw a molecule either passed by its SMILES string or as rdkit.Chem.rdchem.Mol object.
    
    Args:
        mol (string or rdkit.Chem.rdchem.Mol): Molecule to be drawn.
    
    Return:
        None
        
    """
    # Parse input molecule by differing cases
    if isinstance(mol, str):
        # Make a molecule (rdkit.Chem.rdchem.Mol) object from the SMILES string
        mol_obj = rdkit.Chem.MolFromSmiles(mol)
    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        # In this case, the input is already a (rdkit.Chem.rdchem.Mol) molecule object
        mol_obj = mol
    else:
        err_msg = f"The input molecule must be either a string or a 'rdkit.Chem.rdchem.Mol' object, got type '{type(mol)}' instead."
        raise TypeError(err_msg)
    
    # Display the molecule
    display(rdkit.Chem.Draw.MolToImage(mol_obj))

class FingerprintMapper(object):
    """
    Generate a mapper from SMILES strings to (molecular) fingerprints.
    """
    def __init__(self, 
                 smiles_list, 
                 fp_radius=2, 
                 fp_nBits=2048, 
                 fp_dtype=np.float64):
        """
        Args:
            smiles_list (list): List of smiles strings.
            fp_radius (int): Radius of fingerprints.
                (Default: 2)
            fp_nBits (int): Number of bits for fingerprints.
                (Default: 2048)
            fp_dtype (np.dtype): Data type for the entries of the fingerprints.
                Note that the entries of the fingerprints are binary.
                However, for applications were these fingerprints are used as inputs
                to a model, it might be necessary to represent them as floats.
                (Default: np.float64)
        """
        # Assign inputs to class attributes
        self.smiles_list = list(set(smiles_list)) # Remove duplicates
        self.fp_radius   = fp_radius
        self.fp_nBits    = fp_nBits

        # Initialize certain class attributes to None that will be generated below
        self._smiles_to_mol_obj_map = None
        self._smiles_to_fp_map      = None

        # Generate a dictionary mapping SMILES string to corresponding molecular objects
        self._generate_smiles_to_mol_obj_map()

        # Generate a dictionary mapping SMILES string to corresponding fingerprint
        self._generate_smiles_to_fp_map()
    
    @property
    def tag(self):
        """ Return the tag (i.e. label) for the fingerprint mapper. """
        # The tag contains the fingerprint type [i.e. Morgan fingerprint (MFP)], 
        # fingerprint radius, and the number of bits of the fingerprint.
        return f"MFP{self.fp_radius}@{self.fp_nBits}"

    def _generate_smiles_to_mol_obj_map(self):
        """ Generate a dictionary mapping SMILES strings to molecular objects. """
        print(f"Generate dictionary mapping SMILES strings to corresponding molecular objects...")
        start_time = time.time()
        self._smiles_to_mol_obj_map = {smiles: self._get_molecule_object_from_smiles(smiles) for smiles in self.smiles_list}
        print(f"Done. Duration: {time.time()-start_time: .1f} s\n")

    def set_fp_params(self, 
                      new_fp_radius=None, 
                      new_fp_nBits=None):
        """
        Set the fingerprint number of bits as well as radiu and re-generate the map 
        from SMILES string to corresponding fingerprint. 

        Args:
            new_fp_radius (int or None): Radius of fingerprints to be set if not None.
                (Default: None)
            new_fp_nBits (int or None): Number of bits for fingerprints to be set if not None.
                (Default: None)
        """
        if radius is not None:
            self.fp_radius = int(new_fp_radius)

        if nBits is not None:
            self.fp_nBits = int(new_fp_nBits)

        self._generate_smiles_to_fp_map()

    def set_nBits(self, 
                  new_fp_nBits):
        """ 
        Set the fingerprint number of bits and re-generate the map from SMILES string to corresponding fingerprint. 
        
        Args:
            new_fp_nBits (int or None): Number of bits for fingerprints to be set if not None.
                (Default: None)
        """
        self.fp_nBits = int(new_fp_nBits)
        self._generate_smiles_to_fp_map()

    def set_radius(self, 
                   new_fp_radius):
        """ 
        Set the fingerprint radius and re-generate the map from SMILES string to corresponding fingerprint. 
        
        Args:
            new_fp_radius (int or None): Radius of fingerprints to be set if not None.
                (Default: None)
        """
        self.fp_radius = int(new_fp_radius)
        self._generate_smiles_to_fp_map()

    def _generate_smiles_to_fp_map(self):
        """ Generate a dictionary mapping SMILES strings to corresponding fingerprints. """
        # Check that 'self._smiles_to_mol_obj_map' has been generated before (i.e., is not None any longer)
        if self._smiles_to_mol_obj_map is None:
            err_msg = f"Cannot generate the mapping from SMILES strings to fingerprints, please call the method '_generate_smiles_to_mol_obj_map' first."
            raise AttributeError(err_msg)

        print(f"Generate dictionary mapping SMILES strings to corresponding fingerprints...")
        start_time = time.time()
        self._smiles_to_fp_map = {smiles: tuple(self._get_fingerprint_from_molecule_object(mol_obj)) for smiles, mol_obj in self._smiles_to_mol_obj_map.items()}
        print(f"Done. Duration: {time.time()-start_time: .1f} s\n")

    def __call__(self, 
                 smiles):
        """ 
        Return the fingerprint (as numpy array) of the input SMILES string 
        (if a fingerprint has been constructed for this passed SMILES string). 

        Args:
            smiles (str): Smiles string of a molecule.

        Return:
            The fingerprint of the smiles string as numpy array
        
        """
        if smiles in self._smiles_to_fp_map:
            return self._smiles_to_fp_map[smiles]
        else:
            err_msg = f"The input SMILES string '{smiles}' is not one of the expected SMILES strings (passed when initializing the mapper object)."
            raise ValueError(err_msg)

    def _get_fingerprint_from_smiles(self, 
                                     molecule_smiles):
        """
        Return the fingerprint of the input SMILES string. 

        Args:
            molecule_smiles (str): Smiles string of a molecule.

        Return:
            (RDKIT fingerprint object) Fingerprint of the molecule.
        
        """
        # Get the molecular object of the input molecule
        molecule_obj = self._get_molecule_object_from_smiles(molecule_smiles)

        # Generate a molecular fingerprint from the molecular object
        return self._get_fingerprint_from_molecule_object(molecule_obj)

    def _get_molecule_object_from_smiles(self, 
                                         molecule_smiles):
        """
        Return the rdkit.Chem.rdchem.Mol object of the input SMILES string. 
        
        Args:
            molecule_smiles (str): Smiles string of a molecule.

        Return:
            (RDKit molecule object): RDKit object of the molecule.
        
        """
        return rdkit.Chem.MolFromSmiles(molecule_smiles)

    def _get_fingerprint_from_molecule_object(self, 
                                              molecule_obj):
        """ 
        Return the fingerprint of the input rdkit.Chem.rdchem.Mol object. 

        Args:
            molecule_obj (RDKit molecule object): RDKit object of the molecule.

        Return:
            (RDKIT fingerprint object) Fingerprint of the molecule.
        
        """
        return rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule_obj, radius=self.fp_radius, nBits=self.fp_nBits)

    def summary(self):
        """ Display a summary. """
        num_unique_smiles = len(set(self._smiles_to_fp_map.keys()))
        num_unique_fp     = len(set(self._smiles_to_fp_map.values()))
        ratio             = float(num_unique_fp)/float(num_unique_smiles)
        print(f"Fingerprint tag:                   {self.tag}")
        print(f"Fingerprint parameters - radius:   {self.fp_radius}")
        print(f"Fingerprint parameters - nBits:    {self.fp_nBits}")
        print(f"Number of (unique) SMILES strings: {num_unique_smiles}")
        print(f"Number of unique fingerprints:     {num_unique_fp}")
        print(f"Ratio:                             {ratio*100:.4f}%")