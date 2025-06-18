# model_definitions.py

# Import public modules
import collections
import torch
import torch_scatter

# Import custom modules
from . import message_passing_layers
from .. import utils

# Define base molecule model
class MoleculeBaseModel(torch.nn.Module):
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
            (torch.tensor): Latent (vectorial) representation of the (entire) molecule.

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

# Define a GNN for the molecule
class MoleculeGNNModel(MoleculeBaseModel):
    def __init__(self, 
                 config_dict):
        """
        Args:
            config_dict (dict): Config dictionary.
        """
        # Initialize the base class
        super().__init__(config_dict)

        # Define the embeddings for each of the different vertex features
        self.embeddings_dict = torch.nn.ModuleDict(collections.OrderedDict({
            feature_name: torch.nn.Embedding(len(feature_map), self.config_dict['embedding_dim']) for feature_name, feature_map in self.config_dict['vertex_feature_map'].items()
        }))

        # Define the FCNN before the message passing network (acting on the embedded and concatenated vertex features)
        input_dim             = len(self.config_dict['vertex_feature_map'])*self.config_dict['embedding_dim']
        self.fcnn_before_mpnn = utils.define_fcnn('fcnn_before_mpnn', 
                                                  input_dim=input_dim, 
                                                  hidden_params=self.config_dict['fcnn_before_mpnn']['hidden_params'], 
                                                  output_dim=self.config_dict['mpnn']['dim'], 
                                                  activation_fn=self.config_dict['fcnn_before_mpnn']['activation_fn'],
                                                  dropout=self.config_dict['fcnn_before_mpnn']['dropout'])

        # Define the message passing layer
        self.message_passing_layer = message_passing_layers.GINLayer(features_dim=self.config_dict['mpnn']['dim'], 
                                                                     num_hidden=self.config_dict['mpnn']['num_hidden'],
                                                                     activation_fn=self.config_dict['mpnn']['activation_fn'], 
                                                                     dropout=self.config_dict['mpnn']['dropout'],
                                                                     layer_norm_mode=self.config_dict['mpnn']['layer_norm_mode'],
                                                                     **self.config_dict['mpnn']['params'])
        

        # Define the FCNN after the message passing network (acting on the concatenated message passing outputs)
        input_dim            = self.config_dict['num_message_passes']*self.config_dict['mpnn']['dim']
        self.fcnn_after_mpnn = utils.define_fcnn('fcnn_after_mpnn', 
                                                 input_dim=input_dim, 
                                                 hidden_params=self.config_dict['fcnn_after_mpnn']['hidden_params'], 
                                                 output_dim=self.config_dict['output_dim'],
                                                 activation_fn=self.config_dict['fcnn_after_mpnn']['activation_fn'],
                                                 dropout=self.config_dict['fcnn_after_mpnn']['dropout'])

        # Extract the vertex aggregation using 'sum' if not specified in configurations and check its validity
        self.vertex_aggregation = self.config_dict.get('vertex_aggregation', 'sum')
        if self.vertex_aggregation not in ['sum', 'mean']:
            err_msg = f"The 'vertex_aggregation' (molecule model) must be 'sum' or 'mean', got '{self.config_dict['vertex_aggregation']}' instead."
            raise ValueError(err_msg)

        # Define the vertex aggregation function used to aggregate the vertex features over all vertices of each graph
        # thereby generating the (aggregated) features of each graph
        # Remarks: (1) Aggregation is done over the datapoints axis (dim=0)
        #          (2) 'torch_scatter.scatter_<?>' is not deterministic for GPUs/CUDA because the elements (to which the scatter 'maps' to)
        #              will be assigned randomly to threads that then leads to a random order of the 'reduced'/processed elements.
        #              Although some reduction operations (e.g. sum, mean etc.) are invariant under permutations of the elements (and thus 
        #              the order of these elements), due to floating point arithmetics, the result depends on the order of the elements.
        #              => Thus 'torch_scatter.scatter_<?>' might result in 'non-reproducible' results for GPUs/CUDA (because the order of
        #                 elements can be random for different runs of the same operation)
        #              => Using 'torch_scatter.segment_csr', the element order is fixed thereby leading to 'reproducible' results
        #                 See for example: https://github.com/rusty1s/pytorch_scatter/issues/226
        #          (3) The 'reproducible sum/mean-aggregation' below "torch_scatter.segment_csr(..., reduce='mean' or 'sum')" is equivalent 
        #              to the following 'non-guaranteed reproducible sum/mean-aggregations':
        #              self.vertex_aggregation_fn = lambda x, data_batch : torch_scatter.scatter_sum(x, data_batch, dim=0)
        #              self.vertex_aggregation_fn = lambda x, data_batch : torch_scatter.scatter_mean(x, data_batch, dim=0)
        self.vertex_aggregation_fn = lambda x, data_ptr : torch_scatter.segment_csr(x, data_ptr, reduce=self.vertex_aggregation)

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        return self.config_dict['output_dim']

    def forward(self, 
                data):
        """ 
        Define the forward pass.

        Args:
            data (torch.data): Torch data object.
        
        Return:
            (torch.tensor): Latent (vectorial) representation of the (entire) molecule.

        """
        # Determine and return the (vectorial) latent representation of the (entire) molecule 
        return self.get_x_molecule(data)

    def get_x_molecule(self, 
                       data):
        """ 
        Return the the (vectorial) latent representation of the (entire) molecule for the input data.

        Args:
            data (torch.data): Torch data object.
        
        Return:
            (torch.tensor): (Vectorial) latent representation of the (entire) molecule.

        """
        # Get the latent representation of the molecular graph vertices (i.e. the post-processed vertex-feature-set output of the MPNN-module)
        x_molecule_vertices = self.get_x_molecule_vertices(data)

        # Aggregate the vertex features over each molecular graph to combine them to the latent molecule (graph) representation
        return self.vertex_aggregation_fn(x_molecule_vertices, data.ptr)

    def get_x_molecule_vertices(self, 
                                data):
        """
        Return the latent representation of the molecular graph vertices.
        
        Args:
            data (torch.data): Torch data object.
                
        Return:
            (torch.tensor): Latent representations of molecular graph vertices as tensor.
        
        """
        # Embed the different vertex features creating a list of their embeddings
        # Remark: The features (i.e. feature types) are prefixed with a "x_<feature_name>" in the data.
        embeddings_list = [embedding( getattr(data, f"x_{feature_name}") ) for feature_name, embedding in self.embeddings_dict.items()]

        # Concatenate these embeddings along the features axis (dim=1)
        x = torch.cat(embeddings_list, dim=1)

        # Reduct this dimension from #vertex_feature_types*embedding_dim -> MPNN_dim (dimension of the MPNN features)
        x = self.fcnn_before_mpnn(x)

        # Perform multiple message-passing steps and construct a list from the outputs
        message_passing_outputs = list()
        for _ in range(self.config_dict['num_message_passes']):
            # Pass messages
            x = self.message_passing_layer(x, data.edge_index, data.batch)

            # Append the output to the corresponding list
            message_passing_outputs.append(x)            

        # Concatenate the message passing outputs along the features axis (dim=1)
        x = torch.cat(message_passing_outputs, dim=1)

        # Reduct this dimension from #MP_steps*MPNN_dim -> output (dimension of the output) and return the result
        return self.fcnn_after_mpnn(x)
