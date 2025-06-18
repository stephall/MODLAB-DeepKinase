# model_storage.py

# Import public modules
import os
import re
import torch
import pickle
import shutil
from pathlib import Path

class ModelStorage(object):
    def __init__(self, 
                 checkpoints_dir_path):
        """
        Args:
            checkpoints_dir_path (str): Path to the directory the checkpoints 
                are (or should be) stored in.
        """
        # Assign inputs to class attributes
        self.checkpoints_dir_path = checkpoints_dir_path
    
    def cleanup_checkpoints_dir(self):
        """ Delete all folders and files in the checkpoints directory. """
        # The checkpoints directory can only be cleaned in case it already exists
        if os.path.isdir(self.checkpoints_dir_path):
            # Loop over all files in the directory and delete/remove them
            for file_name in os.listdir(self.checkpoints_dir_path):
                # Generate the path to the current file
                file_path = Path(self.checkpoints_dir_path, file_name)

                # Differ cases where the file is actually a file or a folder
                if os.path.isdir(file_path):
                    # The file is actually a directory, so remove it and its content
                    shutil.rmtree(file_path)
                else:
                    # The file is an actual file, so remove it
                    os.remove(file_path)

    @property
    def _model_params_file_name(self):
        """ Return the name of the files that hold the parameters of the model (function). """
        return f'model_parameters.pt'

    def _get_tracked_list_dict_file_name(self):
        """ Return the file name of a file containing the dictionary with the list of the tracked quantities. """
        return f'tracked_quantities.pickle'

    def _get_epoch_dir_name(self, 
                            epoch):
        """
        Return the name of the directory corresponding to the passed epoch. 

        Args:
            epoch (int): Epoch for which the directory name should be returned for.

        Return:
            (str) Directory name as string.
    
        """
        return f'epoch_{epoch}'

    @property
    def _epoch_dir_naming_pattern(self):
        """
        Return the naming pattern of the epoch directories.
        Must match with what is defined in '_get_epoch_dir_name'.
        """
        return r'epoch_(\d*)'

    def _get_model_params_file_path(self, 
                                    epoch):
        """ 
        Return the path to the files that hold the parameters of the model (function) for the input epoch. 

        Args:
            epoch (int): Epoch for which the path to the parameter file should be returned for.
        
        Return:
            (pathlib.Path object): Path to the model parameters file. 
        
        """
        # Get the path to the directory corresponding to the epoch
        epoch_dir_path = self._get_epoch_dir_path(epoch)

        # Generate the file path to the model parameters and return it
        return Path(epoch_dir_path, self._model_params_file_name)

    def _get_tracked_list_dict_file_path(self, 
                                         epoch):
        """
        Return the path to the tracked list dictionary file of the input epoch. 
        
        Args:
            epoch (int): The epoch the tracked list dictionary corresponds to.

        Return:
            (pathlib.Path object): Path to the tracked list dictionary file.
    
        """
        # Get the path to the directory corresponding to the epoch
        epoch_dir_path = self._get_epoch_dir_path(epoch)

        # Get the file name of the tracked list dictionary
        tracked_list_file_name = self._get_tracked_list_dict_file_name()

        # Generate the file path to the tracked list and return it
        return Path(epoch_dir_path, tracked_list_file_name)

    def _get_epoch_dir_path(self, 
                            epoch):
        """ 
        Return the path to the directory corresponding to the passed epoch. 
        
        Args:
            epoch (int): The epoch for which the path to its directory 
                should be returned.

        Return:
            (pathlib.Path object): Path to the epoch directory.
        
        """
        # Get the directory name of the epoch
        epoch_dir_name = self._get_epoch_dir_name(epoch)

        # Construct the directory path
        epoch_dir_path = str( Path(self.checkpoints_dir_path, epoch_dir_name) )

        # Create the file directory 'epoch_dir_path' if it doesn't exist yet
        if not os.path.isdir(epoch_dir_path):
            os.makedirs(epoch_dir_path)

        # Return the path to this directory
        return epoch_dir_path

    def save_model_parameters(self, 
                              model, 
                              epoch):
        """
        Save the model (function) parameter for the passed epoch. 
        
        Args:
            model (object): Model (function) as pytorch model object.
            epoch (int): Epoch for which the parameters of the model function should be saved.

        Return:
            None
        """
        # Get the filepath under which we want to save the model function's parameters
        model_params_file_path = self._get_model_params_file_path(epoch)

        # Store the model parameters
        torch.save(model.state_dict(), model_params_file_path)

    def save_tracked_list_dict(self, 
                               tracked_list_dict, 
                               epoch):
        """
        Save the dictionary containing the tracked lists for the input epoch. 
        
        Args:
            tracked_list_dict (dict): Dictionary containing the tracked lists of form [(epoch, tracked),...]
                as values to the keys of tracked_list_dict (corresponding to the tracked quantities).
            epoch (int): Epoch for which the parameters of the model function should be saved.

        Return:
            None
        """
        # Get the file path under which the dictionary, containing the list of tracked quantities, 
        # should be saved in
        tracked_list_dict_file_path = self._get_tracked_list_dict_file_path(epoch)

        # Store the dictionary as pickle file
        with open(tracked_list_dict_file_path, 'wb') as file:
            pickle.dump(tracked_list_dict, file)

    def load_model_parameters(self, 
                              model, 
                              epoch='last'):
        """
        Load the model parameters saved for a certain epoch.

        Args:
            model (object): Model (function) as pytorch model object.
            epoch (int or str): Epoch to load the model from. This is either a positive integer or a str 'last', 
                which will return the parameters of the last epoch in 'self.save_dir_path'.
                (Default: 'last')

        Return:
            None
        
        """
        # Parse the input epoch.
        # In case the passed epoch is an integer, this method will check if a directory to this epoch 
        # exists (in which case no error or exception is thrown) and the passed epoch is returned.
        # In case the passed epoch is 'last', this method will select the last epoch and return it
        # (unless there is no epoch dictionary in which case it will throw an error or exception).
        epoch = self.parse_epoch(epoch)

        # Get the filepath under which we want to save the model (function) parameters
        model_params_file_path = self._get_model_params_file_path(epoch)

        # Check that this file exists
        if not os.path.isfile(model_params_file_path):
            err_msg = f"Can't load model parameters for model '{self.label}' for epoch {epoch} as the paremeters have not been saved for this epoch."
            raise FileNotFoundError(err_msg)

        # Get the device the model is on
        device = model.device

        # Load the model parameters, while mapping them to the device the model is on
        model.load_state_dict( torch.load(model_params_file_path, map_location=model.device) )

    def get_tracked_list_dict(self, 
                              epoch='last'):
        """
        Return the tracked list dictionary saved for a certain epoch.

        Args:
            epoch (int or str): Epoch to load the model from. This is either a positive integer or a str 'last', 
                which will return the parameters of the last epoch in 'self.model_params_dir'.
                (Default: 'last')

        Return:
            (dict): Dictionary containing the tracked lists as values to the tracked quantities.
        
        """
        # Try to select the input epoch and if it worked (not errors/exceptions are thrown)
        epoch = self.parse_epoch(epoch)

        # Get the file path under which the dictionary, containing the lists of tracked quantities,
        # should be saved in
        tracked_list_dict_file_path = self._get_tracked_list_dict_file_path(epoch)

        # Load dictionary from the pickle file and return it
        with open(tracked_list_dict_file_path, 'rb') as file:
            tracked_list_dict = pickle.load(file)

        return tracked_list_dict

    def parse_epoch(self, 
                    epoch):
        """
        Parse the input epoch.
        # In case the passed epoch is an integer, this method will check if a directory to this epoch 
        # exists (in which case no error or exception is thrown) and the passed epoch is returned.
        # In case the passed epoch is 'last', this method will select the last epoch and return it
        # (unless there is no epoch dictionary in which case it will throw an error or exception).

        Args:
            epoch (int or str): The epoch tried to be selected. This is either a positive integer or a 
                str 'last', which will return the parameters of the last epoch in 'self.model_params_dir'.
                (Default: 'last')

        Return:
            None
        
        """
        # Check if epoch is a positive integer or a str 'last'
        if isinstance(epoch, str):
            if epoch!='last':
                err_msg = f"The input 'epoch' must be 'last' if it is passed as a string, got '{epoch}' instead."
                raise ValueError(err_msg)

        elif isinstance(epoch, int):
            if epoch<0:
                err_msg = f"The input 'epoch' must be positive if it is passed as an integer, got '{epoch}' instead."
                raise ValueError(err_msg)
        else:
            err_msg = f"The input 'epoch' must a positive integer or a str 'last', got type '{type(epoch)}' instead."
            raise TypeError(err_msg)

        # Differ cases where epoch is 'last' or not
        if epoch=='last':
            # Find and return the last epoch
            return self._find_last_epoch()
        else:
            # Construct the name of the epoch directory
            epoch_dir_name = self._get_epoch_dir_name(epoch)

            # Check if this file name exists in the directory 'self.save_dir_path'
            if epoch_dir_name not in os.listdir(self.checkpoints_dir_path):
                err_msg = f"Can't load for the model as nothing has been saved for epoch {epoch} for this model."
                raise ValueError(err_msg)

            # Return the passed epoch
            return epoch

    def _find_last_epoch(self):
        """ Find the last saved epoch of a model and return it. """
        # Get all files in the directory 'self.save_dir_path'
        file_names = os.listdir(self.checkpoints_dir_path)

        # Initialize the maximal epoch to -1
        max_epoch = -1

        # Loop over all the file names
        for file_name in file_names:
            # The parameter files are named according to the pattern 'epoch_<epoch>'
            naming_pattern = self._epoch_dir_naming_pattern
            findall_obj    = re.findall(naming_pattern, file_name)
            
            # Only one item should have been found, which corresponds to the string of the epoch 
            if len(findall_obj)==1:
                # Make the found epoch an integer
                epoch = int(findall_obj[0])

                # Update the maximal epoch
                max_epoch = max([epoch, max_epoch])

        # In case max_epoch is still -1, no model parameters have been saved
        if max_epoch==-1:
            err_msg = f"Can't load for model '{self.label}' as nothing has been saved for this model."
            raise ValueError(err_msg)

        return max_epoch