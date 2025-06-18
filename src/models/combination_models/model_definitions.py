# model_definitions.py

# Import public modules
import torch
import torch_scatter
import torch_geometric

# Define base combination model
class CombinationBaseModel(torch.nn.Module):
    def __init__(self, 
                 config_dict, 
                 x_molecule_dim, 
                 x_protein_dim):
        """
        Args:
            config_dict (dict): Config dictionary.
            x_molecule_dim (int): The dimension of the latent space features of a molecule.
            x_protein_dim (int): The dimension of latent space features of a protein.
        """
        # Initialize the base class
        super().__init__()

        # Assign inputs to class attributes
        self.config_dict    = config_dict
        self.x_molecule_dim = x_molecule_dim
        self.x_protein_dim  = x_protein_dim

    # Define virtual methods that have to be overloaded
    def forward(self, 
                x_molecule, 
                x_protein):
        """ 
        Define the forward pass.

        Args:
            x_molecule (torch.tensor): The latent space features of a molecule.
            x_protein (torch.tensor): The latent space features of a protein.
        
        Return:
            (torch.tensor): Combined latent space features.

        """
        raise NotImplementedError("Any child class must implement the method 'forward'.")

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        raise NotImplementedError("Any child class must implement the (property) method 'output_dim'.")

    @property
    def expected_molecule_latent_representation(self):
        """ Return the expected molecule latent representation as string. """
        raise NotImplementedError("Any child class must implement the (property) method 'expected_molecule_latent_representation'.")
    
    def display_model_information(self, 
                                  **kwargs):
        """
        Display model information.

        Remark: This method might for example be called when displaying metric values during training.
        
        """
        # A child class can implement this method to display some information.
        # By default, no information is displayed
        pass

class ConcatenationModel(CombinationBaseModel):
    def __init__(self, 
                 config_dict, 
                 x_molecule_dim, 
                 x_protein_dim):
        """
        Args:
            config_dict (dict): Config dictionary.
            x_molecule_dim (int): The dimension of the latent space features of a molecule.
            x_protein_dim (int): The dimension of latent space features of a protein.
        """
        # Initialize the base class
        super().__init__(config_dict, x_molecule_dim, x_protein_dim)

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        return self.x_protein_dim + self.x_molecule_dim

    @property
    def expected_molecule_latent_representation(self):
        """ This model expects a vectorial latent representation of the molecule named 'x_molecule'. """
        return 'x_molecule'

    def forward(self, 
                x_molecule, 
                x_protein):
        """ 
        Define the forward pass.

        Args:
            x_molecule (torch.tensor): The latent features of a molecule.
            x_protein (torch.tensor): The latent features of a protein.
        
        Return:
            (torch.tensor): Combined latent features.

        """
        # Concatenate the molecule and protein features along their feature axis (dim=1)
        return torch.cat( [x_molecule, x_protein], dim=1 )
