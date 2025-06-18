#model_definitions.py

# Import public modules
import torch

# Import custom modules
from .. import utils

# Define base protein model
class ProteinBaseModel(torch.nn.Module):
    def __init__(self, 
                 config_dict):
        """
        Args:
            config_dict (dict): Config dictionary.
        """
        # Initialize the base class
        super().__init__()

        # Assign the config dictionary to a class attribute
        self.config_dict = config_dict

    def forward(self, 
                data):
        """ 
        Define the forward pass.

        Args:
            data (torch.data): Torch data object.
        
        Return:
            (torch.tensor): Latent features of the protein.

        """
        raise NotImplementedError("Any child class must implement the method 'forward'.")

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        raise NotImplementedError("Any child class must implement the (property) method 'output_dim'.")

    def display_model_information(self, 
                                  **kwargs):
        """
        Display model information.

        Remark: This method might for example be called when displaying metric values during training.
        
        """
        # A child class can implement this method to display some information.
        # By default, no information is displayed
        pass

class ProteinIndexFCNNModel(ProteinBaseModel):
    def __init__(self, 
                 config_dict):
        """
        Args:
            config_dict (dict): Config dictionary.
        """
        # Initialize the base/super/parent class
        super().__init__(config_dict)

        # Check that the protein featurization is of type 'protein_index'
        # and throw an error if it isn't
        if not self.config_dict['protein_representation_params']['type']=='protein_index':
            err_msg = f"The model 'ProteinIndexFCNNModel' requires 'protein_index' as protein featurization type, got type '{self.config_dict['protein_representation_params']['type']}' instead."
            raise ValueError(err_msg)

        # Initialize an embedding for the protein indices
        num_proteins = len( self.config_dict['protein_representation_params']['protein_id_to_protein_index_map'] )
        self.embedding = torch.nn.Embedding(num_proteins, self.config_dict['embedding_dim'])

        # Define a FCNN model
        self.fcnn = utils.define_fcnn('protein', 
                                      input_dim=self.config_dict['embedding_dim'], 
                                      hidden_params=self.config_dict['hidden_params'], 
                                      output_dim=self.config_dict['output_dim'],
                                      activation_fn=self.config_dict['activation_fn'],
                                      dropout=self.config_dict['dropout'])

    def forward(self, 
                data):
        """ 
        Define the forward pass.

        Args:
            data (torch.data): Torch data object.
        
        Return:
            (torch.tensor): Latent features of the protein.

        """
        # First embed the protein features
        x = self.embedding(data.protein_features)

        # Pass the embedded protein features through the FCNN model
        return self.fcnn(x)

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        return self.config_dict['output_dim']