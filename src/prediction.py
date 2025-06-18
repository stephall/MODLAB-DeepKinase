# molecules_prediction.py

# Import public modules
import collections
import torch
import torch_geometric
import tqdm
import numpy as np
import pandas as pd

# Import custom modules
from src import data_handling
from src import random_handling
from src import utils

class Predictor(object):
    def __init__(self, 
                 model, 
                 model_config_dict,
                 nswcs_to_graph_map, 
                 batch_size=128):
        """
        Args:
            model (model object): Model object derived from torch.nn.Module
                with additional custom methods.
            model_config_dict (dict): Configurations dictionary of the model.
            nswcs_to_graph_map (dict): Dictionary mapping non-stereochemical 
                washed canonical SMILES (nswcs) strings to a dictionary
                representing the molecular graph of the nswcs's molecule.
            batch_size (int): Batch size to be used during prediction
                (Default: 128).
            
        """

        # Extract the protein representation parameters from the model configurations dictionary
        protein_representation_params = model_config_dict['protein_model']['protein_representation_params']
        
        # Only 'protein_index' is implemented as protein representation type
        if protein_representation_params['type']!='protein_index':
            raise NotImplementedError(f"Predictions are not implemented for protein representations of type '{protein_representation_params['type']}'.")

        # Assign inputs to attributes
        self.model = model
        self.nswcs_to_graph_map = nswcs_to_graph_map
        self.protein_id_to_protein_index_map = protein_representation_params['protein_id_to_protein_index_map']
        self.batch_size = batch_size

        # Define a radom handler
        self.random_handler = random_handling.RandomHandler()
    
    def run(self,
           random_seed=42):
        """
        Predict and return a pandas DataFrame with the predictions.

        Args:
            random_seed (int): Random seed to be set for prediction.
                This ensures deterministic prediction (in case of
                probabilistic predictions).
                (Default: 42)

        Return:
            (pandas.DataFrame): Pandas DataFrame holding the predictions.
        """
        # Construct the dataloader
        dataloader = self.get_dataloader(device=self.model.device)

        # Define data attributes that should be added to the prediction
        data_attributes = ['nswcs', 'protein_id']
        
        # Store the initial setting if the model is training or not 
        # Remark: 'self.model.training' is a boolean
        initial_model_training_flag = self.model.training
        
        # Set the model in evaluation mode
        self.model.eval()
        
        # Set the random seed for predictions using the evaluation random handler
        self.random_handler.set_seed(random_seed)
        
        # Loop over the batches of the set
        quantities_dict = collections.defaultdict(list)
        with torch.no_grad(): # Don't need to use gradients
            for batch_data in tqdm.tqdm(dataloader, total=len(dataloader)): # Loop over all batches
                ### Step 1: Extract certain data attributes
                for data_attribute in data_attributes:
                    # Remark: If 'data_attribute' is not an attribute of batch_data, 'getattr' will throw an
                    #         AttributeError that is caught here and rethrown using a custom error message
                    try:
                        batch_data_attribute_values = getattr(batch_data, data_attribute)
                    except AttributeError:
                        # Determine all attributes of the batch data object
                        batch_data_attributes = re.findall(r'(\w*?)=', str(batch_data))
                        err_msg = f"The passed data attribute '{data_attribute}' is not an attribute of the torch batch data objects for the '{set_name}' set.\nPlease use one of the following attributes:\n{batch_data_attributes}"
                        raise AttributeError(err_msg)
                    
                    # Cast the attribute values of the batch to a numpy array
                    if torch.is_tensor(batch_data_attribute_values):
                        batch_data_attribute_values = batch_data_attribute_values.cpu().detach().numpy()
                    else:
                        batch_data_attribute_values = np.array(batch_data_attribute_values)
                        
                    # Append the attribute values of the batch to the corresponding list
                    quantities_dict[data_attribute].append(batch_data_attribute_values)
                
                ### Step 2: Predict the model output for the current batch
                batch_prediction = self.model(batch_data)
                batch_prediction = batch_prediction.cpu().detach().numpy()
                if batch_prediction.ndim==0: # If scalar, make it 1D array with single entry
                    batch_prediction = batch_prediction.reshape(1)
        
                # Add the batch prediction values to corresponding list
                quantities_dict['prediction'].append(batch_prediction)
                
        # Reset the random states to initial values before prediction
        self.random_handler.reset_states()
        
        # Set the model back to training mode if it was initially in training mode 
        # (and thus the boolean flag 'initial_model_training_flag' is true)
        if initial_model_training_flag:
            # If the initial model training flag was to true, set the model into training mode
            self.model.train()
            
        # Stack the numpy arrays in the lists corresponding to the dictionary-keys of quantities_dict
        # and return the resulting dictionary
        return pd.DataFrame({key: np.concatenate(value) for key, value in quantities_dict.items()})

    def get_dataloader(self, 
                       device=None):
        """
        Return a dataloader objects to be used for prediction

        Args:
            device (torch.device or None): Device the datasets (for each train, valid, test) should be moved to
                or None (in which case the device will be CUDA if it is available and otherwise CPU).
                (Default: None)
            
        Return:
            (dataloader object): Either a torch_geometric.loader.DataLoader or a torch.utils.data.DataLoader 
                dataloader object to be used for prediction

        """
        # In case that the device is not passed (is None), use CUDA if available and CPU otherwise
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate a list containing a torch data objects for each protein-molecule pair (pmp)
        torch_data_list = self.generate_torch_data_list()

        # Generate a ListDataset from the torch data list
        torch_dataset = data_handling.ListDataset(torch_data_list, device=device)

        # Determine the appropriate DataLoader by differing the cases where the torch_dataset is a subclass
        # of 'torch_geometric.data.dataset.Dataset' (so graph structured data) or not
        if issubclass(type(torch_dataset), torch_geometric.data.dataset.Dataset):
            # In case the dataset is a subset of 'torch_geometric.data.dataset.Dataset', use the DataLoader of pytorch-geometric
            DataLoader = torch_geometric.loader.DataLoader
        else:
            # Otherwise, use the DataLoader of pytorch (non-geometric)
            DataLoader = torch.utils.data.DataLoader

        # Define the dataloader for the current subset and assign it to the dataloader dictionary
        return DataLoader(torch_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          generator=None,
                          num_workers=0, 
                          worker_init_fn=None)

    def generate_torch_data_list(self):
        """
        Generate a list of torch data objects.

        Return:
            (list): List of torch data objects.
        """
        nswcs_list = list(self.nswcs_to_graph_map.keys())
        nswcs_list.sort()
        
        protein_id_list = list(self.protein_id_to_protein_index_map.keys())
        utils.sort_chembl_id_list(protein_id_list)

        # Generate a list containing a torch data objects for each protein-molecule pair (pmp)
        torch_data_list = list()
        for nswcs in nswcs_list:
            for protein_id in protein_id_list:
                pmp_torch_data = self.get_torch_data_for_pmp(nswcs, protein_id)
                torch_data_list.append(pmp_torch_data)

        return torch_data_list
    
    def get_torch_data_for_pmp(self, 
                               nswcs, 
                               protein_id):
        """
        Return the torch data object for the protein-molecule
        pair specified by the inputs.
        
        Args:
            nswcs (str): Non-stereochemical washed canonical 
                SMILES (nswcs) string of a molecule.
            protein_id (str): Protein ChEMBL ID (in the format
                'CHEMBL<ID>') of the protein.

        Return:
            (torch_geometric.data.Data): Torch data object for 
                the input protein-molecule pair (pmp).
        """
        # Get the molecular graph (dictionary) for the current molecule
        molecule_graph_dict = self.nswcs_to_graph_map[nswcs]
    
        # Get the protein representation (dictionary) for the protein id
        protein_repr_dict = {
            'protein_features': self.protein_id_to_protein_index_map[protein_id]
        }
    
        return torch_geometric.data.Data(
            protein_id=protein_id,
            nswcs=nswcs,
            num_nodes=molecule_graph_dict['num_vertices'],
            **{key: torch.tensor(value) for key, value in molecule_graph_dict.items() if key!='num_vertices'},
            **{key: torch.tensor(value) if not isinstance(value, str) else value for key, value in protein_repr_dict.items()}
        )