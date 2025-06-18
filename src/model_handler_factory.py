# model_handler_factory.py

# Import public modules
import os
import logging
import torch
import omegaconf
import hydra
from pathlib import Path

# Import custom modules
from src import utils
from src import data_handling
from src import model_handling
from src import model_handling_rf
from src import model_factory

def define_model_handler(config, 
                         load_data=True,
                         temp_dir_path='../outputs/temp',
                         silent_mode=False):
    """
    Define the model handler and return it together with the (updated) config dictionary. 

    Args:
        config (omegaconf.dictconfig.DictConfig or str): Either an omegaconf-configuration dictionary
            or the path (as string) to a config file that can be loaded and cast to such a dictionary.
        load_data (bool): Should the data be loaded (in addition to the model itself) for the
            model handler? If the model handler requires the training or test set (e.g., for
            performance evaluations), one requires load_data=True. If the model handler is
            only used to obtain the model's prediction (not on the training or test set), one
            can use load_data=False.
            (Default: False)
        temp_dir_path (str): The temporary (output) directory in which temporary files (e.g. log files 
            and figures) should be saved in. This is only used in case a model has already been trained 
            and is loaded for downstream applications.
            (Default: '../outputs/temp')
        silent_mode (bool): Boolean flag if only warnings should be logged by custom loggers and no
            other messages are printed.
            (Default: False)

    Return:
        (model_handling.Model, dict): Defined model handler object and the configuration dictionary
            containing all updates made to the original input 'config' during the model handler defintion.
    
    """
    
    # Unfortunately, some imported (public) modules use the root logger to log their events
    # and because we have to set the root logger level to DEBUG for our events to be logged
    # as INFO or DEBUG, this means that these aforementioned modules will start logging their
    # INFO or DEBUG events. Circumvent this by explicitly setting their logging level to WARN.
    # Remark: This is not an elegant solution but a Hack. The problem is that we should not set
    #         the root loggers level from WARN to DEBUG in the first place, but need to do this here.
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('hydra').setLevel(logging.WARN)
    logging.getLogger('numba').setLevel(logging.WARN) # required if umap module is loaded
    logging.getLogger('PIL').setLevel(logging.INFO)   # Required when drawing molecules using RDKit (logging level of pillow library)

    # Differ cases where the config variable is 'omegaconf.dictconfig.DictConfig' or string
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        if not silent_mode:
            print(f"Loading configurations via hydra\n")

        # The configurations are now accessible via the omegaconf DictConf object 'config' (usually called 'cfg' 
        # when using hydra via 'main_exec(cfg: omegaconf.DictConfig)') by mapping it to a python dictionary
        config_dict = utils.map_dictconfig_to_dict(config)

        # 1-2) Update the path to the raw data files in the config dictionary by mapping it from a relative to an absolute path:
        # 1) Get the relative file path to the raw data base directory
        relative_raw_data_base_dir = config_dict['data_handling']['data_preprocessing']['raw_data_base_dir']

        # 2) Construct the absolute file path to this directory and update the corresponding key-value pair in the config directory with it
        config_dict['data_handling']['data_preprocessing']['raw_data_base_dir'] = str( Path(hydra.utils.get_original_cwd(), relative_raw_data_base_dir) )

        # Define the loggers
        preprocessing_logger = utils.define_logger('./logs/preprocessing.log')
        general_logger       = utils.define_logger('./logs/general.log')
    elif isinstance(config, str):
        if not silent_mode:
            print(f"Loading configurations from the model output folder\n")

        # If config is a string it is assumed to corresponds to the model output directory
        model_output_folder_path = config

        # Check that the a folder exists in the path
        if not os.path.isdir(model_output_folder_path):
            err_msg = f"No model output folder found in {model_output_folder_path}"
            raise FileNotFoundError(err_msg)

        # Get the config dictionary defined in the model output folder
        config_dict = utils.get_config_dict(config_dir_rel_path=model_output_folder_path)

        ########################################################################################################################
        ### Config dictionary updates
        ########################################################################################################################
        # Update the path to the raw data files in the config dictionary
        config_dict['data_handling']['data_preprocessing']['raw_data_base_dir'] = '../raw_data'

        # Define the relative file path to the checkpoint files that are located in the model output folder
        checkpoints_dir_path = str( Path(model_output_folder_path, 'checkpoints') )

        # Update the path to the checkpoints in the config dictionary
        config_dict['training']['checkpoints_dir_path'] = checkpoints_dir_path

        # Create the 'temporary (output) directory' if it doesn't exist
        if not os.path.isdir(temp_dir_path):
            os.makedirs(temp_dir_path)

        # Update the output directory paths for the figures in data preprocessing and training
        # using the 'temporary (output) directory' as the base save directory
        config_dict['data_handling']['data_preprocessing']['figures_dir_path'] = str( Path(temp_dir_path, config_dict['data_handling']['data_preprocessing']['figures_dir_path']) )
        config_dict['training']['figures_dir_path']                            = str( Path(temp_dir_path, config_dict['training']['figures_dir_path']) )

        ########################################################################################################################
        ### Define the loggers
        ########################################################################################################################
        # Remark: Use the 'temporary (output) directory' as the base save directory
        preprocessing_logger = utils.define_logger( str(Path(temp_dir_path, './logs/preprocessing.log')) )
        general_logger       = utils.define_logger( str(Path(temp_dir_path, './logs/general.log')) )
    else:
        err_msg = f"The input 'config' must be either a 'omegaconf.dictconfig.DictConfig' object (obtained when using hydra) or the path to a model output folder as string, got type '{type(config)}' instead.\nThe 'config' is: {config}"
        raise TypeError(err_msg)

    # If 'silent mode' is used, only warnings should be logged
    if silent_mode:
        preprocessing_logger.setLevel(logging.WARN)
        general_logger.setLevel(logging.WARN)

    # Get the device based on CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Log the device and the configs
    general_logger.info(f"Run on device: {device}\n")
    general_logger.info(f"Configs:\n{config_dict}\n")

    # In case the device is not 'cpu', set certain 'global' CUDA settings
    if str(device)!='cpu':
        utils.set_global_cuda_settings(logger=general_logger, make_cuda_operations_deterministic=config_dict['model']['make_cuda_operations_deterministic'])

    # Differ the cases where the data should be loaded or not (in which case there won't be any data for training or 
    # evaluation accessible within the model_handler object!)
    if load_data:
        # Define the data handler (that will itself internally run the data preprocessor)
        data_handler = data_handling.DataHandler(config_dict['data_handling'], preprocessing_logger)
        general_logger.info("Loaded data\n")

        ## Extract some information from the data handler
        # For the molcule model
        if 'molecule_model' in config_dict['model']:
            config_dict['model']['molecule_model']['vertex_feature_map'] = data_handler.data_preprocessor.smiles_to_graph_mapper.vertex_feature_map
            config_dict['model']['molecule_model']['edge_feature_map']   = data_handler.data_preprocessor.smiles_to_graph_mapper.edge_feature_map

        # For the protein model
        if 'protein_model' in config_dict['model']:
            config_dict['model']['protein_model']['protein_representation_params'] = data_handler.data_preprocessor.protein_representation_params

        # For the decision model
        if 'decision_model' in config_dict['model']:
            config_dict['model']['decision_model']['relation_representation_params']                 = data_handler.data_preprocessor.config_dict['relation_representation_params']
            config_dict['model']['decision_model']['decision_layer_configs']['num_train_datapoints'] = data_handler.get_num_datapoints_set('train')
    else:
        data_handler = None
    
    # Define the model handler
    if config_dict['model']['name']=='random_forest':
        ### Case 1: Random forest model
        config_dict['model']['raw_base_data_dir'] = config_dict['data_handling']['data_preprocessing']['raw_data_base_dir']

        # Define the random forest model handler
        model_handler = model_handling_rf.RFModelHandler(data_handler, config_dict=config_dict['model'], logger=general_logger)
    else:
        ### Case 2: All other types of models (i.e., GNN here)
        # Define the model (function) and map it to the device defined above
        model = model_factory.define_model(config_dict['model'], device=device)

        # Define the model handler 
        # Remark: Pass the training configs (e.g. optimizer) as kwargs
        model_handler = model_handling.ModelHandler(model, data_handler, logger=general_logger, **config_dict['training'])
    
    # Return the model handler and the (updated) config dictionary
    return model_handler, config_dict
