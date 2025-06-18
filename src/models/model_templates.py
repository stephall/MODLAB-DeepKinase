#model_templates.py

# Import public modules
import torch
import numpy as np

# Import custom modules
from . import molecule_models
from . import protein_models
from . import combination_models
from . import decision_models

# Define a Base Model on which all model templates will be based on
class BaseModel(torch.nn.Module):
    def __init__(self, config_dict):
        # Initialize the base class
        super().__init__()

        # Assign the inputs to class attributes
        self.config_dict = config_dict

    def display_params(self, 
                       show_values=False):
        """
        Display the parameters of the model. 

        Args:
            show_values (bool): Boolean flag indicating if the
                parameter values should also be shown explicitly.
        
        """
        for name, params in self.named_parameters():
            print('-'*100)
            print(f"{name}: {params.shape}")
            if show_values:
                print(f"Param values: {params}")
                print(f"Sum over the param values: {np.sum(params.cpu().detach().numpy())}")

        print('-'*100)

    @property
    def device(self):
        """ Return the device the model is on. """
        # Get any of the model parameters (here the first one in the iterator 'self.parameters()')
        # and return its device, which will correspond to the device of the model
        return next(self.parameters()).device

# Define a Pair Model template used to construct all pair models (models taking a protein-molecule pair as inputs)
class PairModel(BaseModel):
    def __init__(self, config_dict):
        # Initialize the base class, this will assign the config dictionary to a class attribute 'self.config_dict'
        super().__init__(config_dict=config_dict)

        # Define the different model parts
        self.molecule_model    = molecule_models.define_model(self.config_dict['molecule_model'])
        self.protein_model     = protein_models.define_model(self.config_dict['protein_model'])
        self.combination_model = combination_models.define_model(self.config_dict['combination_model'], self.molecule_model.output_dim, self.protein_model.output_dim)
        self.decision_model    = decision_models.define_model(self.config_dict['decision_model'], self.combination_model.output_dim)
        
    def get_x_combined(self, 
                       data):
        """
        Return the combined representation for the input data.
        
        Args:
            data (torch.data.Data): Data object containing the vertex features 'x' and edge indices as 'edge_index' as attributes.
                data.x (torch.tensor): Vertex features of shape (#vertices, #vertex_features)
                data.edge_index (torch.tensor): Edge indices, representing the adjacency, of shape (2, #edges)
                
        Return:
            (torch.tensor): Combined (molecule and protein) latent representation as tensor.
        
        """
        # Get the latent protein representation
        x_protein = self.get_x_protein(data)

        # Differ cases for the latent molecule representation expected by the combination model
        if self.combination_model.expected_molecule_latent_representation=='x_molecule':
            # Get the (vectorial) latent representation of the (entire) molecule
            x_molecule = self.molecule_model.get_x_molecule(data)

            # Combine the latent representations of the molecule and the protein
            x_combined = self.combination_model(x_molecule, x_protein)
        elif 'x_molecule_vertices':
            # Get the latent representation of the molecular vertices
            x_molecule_vertices = self.molecule_model.get_x_molecule_vertices(data)

            # Combine the latent representations of the molecule and the protein
            x_combined = self.combination_model(x_molecule_vertices, x_protein, data)
        else:
            err_msg = f"The attribute 'expected_molecule_latent_representation' of the combination model must be either 'x_molecule' or 'x_molecule_vertices', got '{self.combination_model.expected_molecule_latent_representation}' instead."
            raise ValueError(err_msg)

        return x_combined

    def get_x_molecule(self, 
                       data):
        """
        Return the (vectorial) latent representation of the (entire) molecule for the input data.
        
        Args:
            data (torch.data.Data): Data object containing the vertex features 'x' and edge indices as 'edge_index' as attributes.
                
        Return:
            (torch.tensor): Latent molecule representation as tensor.
        
        """
        # Determine and return the (vectorial) latent representation of the (entire) molecule 
        return self.molecule_model.get_x_molecule(data)

    def get_x_protein(self, 
                      data):
        """
        Return the latent protein representation for the input data.
        
        Args:
            data (torch.data.Data): Data object containing the vertex features 'x' and edge indices as 'edge_index' as attributes.
                data.x (torch.tensor): Vertex features of shape (#vertices, #vertex_features)
                data.edge_index (torch.tensor): Edge indices, representing the adjacency, of shape (2, #edges)
                
        Return:
            (torch.tensor): Latent protein representation as tensor.
        
        """
        # Feed the data to the protein model to obtain a latent protein representation that is returned
        return self.protein_model(data)

    def get_molecule_vertices_weights_tuple_list(self, 
                                                 data):
        """
        Return a list containing tuples of the vertex weights (i.e. where the tuple index corresponds to the vertex index) for the vertices
        of a molecule conditional on the protein.
        
        Args:
            data (torch.data.Data): Data object containing the vertex features 'x' and edge indices as 'edge_index' as attributes.
                
        Return:
            (list of tuples): List containing tuples of the vertex weights (i.e. where the tuple index corresponds to the vertex index) for the vertices
                of a molecule conditional on the protein.
        
        """
        # Get the latent representation of the molecular vertices
        x_molecule_vertices = self.molecule_model.get_x_molecule_vertices(data)
        
        # Get the latent protein representation
        x_protein = self.get_x_protein(data)

        # Try to get the vertices weights from the combination model and throw an error if the combination model does not have a 
        # method to determine these
        try:
            return self.combination_model.get_molecule_vertices_weights_tuple_list(x_molecule_vertices, x_protein, data)
        except AttributeError:
            err_msg = f"The combination model '{type(self.combination_model)}' cannot determine vertices weights."
            raise AttributeError(err_msg)

    def forward(self, 
                data):
        """
        Define the forward pass.
        
        Args:
            data (torch.data.Data): Data object containing the vertex features 'x' and edge indices as 'edge_index' as attributes.
                data.x (torch.tensor): Vertex features of shape (#vertices, #vertex_features)
                data.edge_index (torch.tensor): Edge indices, representing the adjacency, of shape (2, #edges)
                
        Return:
            (torch.tensor): Model output.
        
        """
        # Get the combined latent space features
        x_combined = self.get_x_combined(data)
        
        # Feed the combined latent representation to the decision model
        y_model = self.decision_model(x_combined)
                
        return y_model

    def loss(self, 
             data):
        """
        Return the loss (torch object) for the input data and current model parameters.

        Args:
            data (torch.data.Data): Data object.

        Return:
            (torch.tensor): Loss value.
            
        """
        # Get the combined latent space features
        x_combined = self.get_x_combined(data)

        # Calculate the loss of the decision model and return it
        return self.decision_model.loss(x_combined, data)

    def metric(self, 
               data):
        """
        Return the metric (torch object) for the input data and current model parameters.

        Args:
            data (torch.data.Data): Data object.

        Return:
            (torch.tensor): Metric value.
            
        """
        # Get the combined latent space features
        x_combined = self.get_x_combined(data)

        # Calculate the metric of the decision model and return it
        return self.decision_model.metric(x_combined, data)

    def forward_up_to_decision_layer(self, 
                                     data):
        """
        Calculate the forward pass up to (but not including) the decision layer. 

        Args:
            data (torch.data.Data): Data object.

        Return:
            (torch.tensor): Output of decision model before the its final 
                (i.e., decision) layer (i.e., what would be passed to the 
                decision layer).
        
        """
        # Get the combined latent space features
        x_combined = self.get_x_combined(data)
        
        # Calculate the forward pass in the decision model up to (but not including) the decision layer
        return self.decision_model.forward_up_to_decision_layer(x_combined)

    def call_decision_layer_method(self, 
                                   method_name, 
                                   data, 
                                   **kwargs):
        """
        Call a decision layer method for the input data and the passed arguments.

        Args:
            method_name (str): Name of the decision layer method.
            data (torch.data.Data): Data object.
            **kwargs (dict): Forwarded to the decision layer method.

        Return:
            (torch.tensor): Output of the decision layer method.
        """
        # Get the combined latent space features
        x_combined = self.get_x_combined(data)

        # Call the decision layer method and return the result
        return self.decision_model.call_decision_layer_method(method_name, x_combined, **kwargs)
            
    def display_model_information(self, 
                                  **kwargs):
        """
        Display model information.

        Remark: This method might for example be called when displaying metric values during training.  

        Args:
            **kwarg: Forwarded to all the different sub-models.

        """
        # Display the model information of all sub-models
        self.molecule_model.display_model_information(**kwargs)
        self.protein_model.display_model_information(**kwargs)
        self.combination_model.display_model_information(**kwargs)
        self.decision_model.display_model_information(**kwargs)