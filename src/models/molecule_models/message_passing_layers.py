#message_passing_layers.py

# Import public modules
import torch
import torch_sparse
import torch_geometric

# Import custom modules
from .. import utils

class GINLayer(torch.nn.Module):
    def __init__(self, 
                 features_dim, 
                 num_hidden=0, 
                 activation_fn='ReLU', 
                 dropout=None, 
                 layer_norm_mode='node', 
                 **kwargs):
        """
        GIN Convolution consists of the following steps:
        AGGREGATION: m_i = sum_{j in N(i)} x_j
        UPDATE:     x'_i = Network((1+eps)*x_i + m_i)

        Remark: Keep the features dimension constant in message passing.

        Args:
            features_dim (int): Feature dimension.
            num_hidden (int): Number of hidden layers.
                (Default: 0) 
            activation_fn (str): Activation function to be used.
                (Default: 'ReLU')
            dropout (float or None): Should dropout be used (dropout probability passed as float) 
                or not (None).
                (Default: None)
            layer_norm_mode (str or None): Which layer norm should be computed ('node' or 'graph')
                or should no layer norm be used (None or 'None')?
                (Default: 'node')
            **kwargs (dict): Additional key-word arguments, such as 'eps' and 'train_eps'.

        """
        # Initialize the base class
        super().__init__()

        # Define the network used in the updating step as the 'Network'
        update_network = utils.define_fcnn('GINlayer', 
                                           input_dim=features_dim, 
                                           hidden_params={'hidden_dims': [features_dim]*num_hidden}, 
                                           output_dim=features_dim,
                                           activation_fn=activation_fn,
                                           dropout=dropout)

        # Extract some parameters from the key-word arguments dictionary and assign 
        # default values if they are not found
        eps       = kwargs.get('eps', 0.0) # (Initial) value for epsilon
        train_eps = kwargs.get('train_eps', False) # Should epsilon be trainable?

        # Initialize the base class
        self.gin_conv = torch_geometric.nn.conv.GINConv(nn=update_network, eps=eps, train_eps=train_eps)

        # Define the layer normalization
        if (layer_norm_mode is None) or (layer_norm_mode=='None'):
            # If the layer_norm_mode is None (or 'None') do not use layer normalization and thus define 'self.layer_norm' 
            # as identify function over the vertex features
            self.layer_norm = lambda x, batch: x
        elif layer_norm_mode in ['graph', 'node']:
            # Apply normalization of the vertex features over either the vertices of each graph (mode='graph') or over each vertex individuall (mode='node')
            self.layer_norm = torch_geometric.nn.LayerNorm(in_channels=features_dim, mode=layer_norm_mode)
        else:
            err_msg = f"The input 'layer_norm_mode' must be either None (or 'None'), 'graph', or 'node', got value '{layer_norm_mode}' instead."
            raise ValueError(err_msg)

    def forward(self, 
                x, 
                edge_index, 
                batch):
        """
        Define the forward pass.
        
        Args:
            x (torch.tensor): Input vertex/node feature matrix of shape (#vertices, #input_features).
            edge_index (torch.tensor): Edge indices.
            batch (torch.tensor): Map that assigns vertices to their corresponding graph.
        
        Return:
            x (torch.tensor): Output vertex/node feature matrix of shape (#vertices, #output_features).            

        Remark: The method 'propagate(self, edge_index: Adj, size: Size = None, **kwargs)' of the class 'torch_geometric.nn.MessagePassing' 
                (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html#MessagePassing) 
                differs the cases where edge_index is either a sparse matrix/tensor (-> Case 1) or not (-> Case 2).
                Case 1 (edge_index is a sparse matrix/tensor):
                    The virtual method 'message_and_aggregate' (to be implemented in specific child/derived class) will be called that
                    does message passing and aggregation in one step and is in most cases implemented using sparse-matrix multiplication.
                    Using matrix multiplication for the convolution, the order of vertices is fixed in any convolution as long as the
                    matrices (adjacency and vertex features) are constructed in a deterministic manner.
                Case 2 (edge_index is a sparse matrix/tensor):
                    First the messages will be created using the method 'message' (usually overriden by child/derived class) and after 
                    these are constructed, they will be aggregated using the method 'aggregate' (might be overriden by child/derived class).
                    As the order of the messages in the resulting iterable of messages might not be deterministic on a GPU (or for multi-processing
                    in general) due to distribution of operations to different threads, the aggregation might lead to non-deterministic results
                    due to floating point arithmetics (as the order of e.g. summed floating points matters for the result of the sum).
                    
                Implications:
                    (*) Using for example 'torch_geometric.nn.conv.GINConv.forward(x, edge_index)' can lead to non-deterministic results on GPUs
                    if edge_index is a (non-sparse) torch tensor of shape (2, #edges) containing the edge pairs.

                    (*) Thus to achieve deterministic results, such a (non-sparse) edge_index tensor should be transformed to a sparse adjacency
                    matrix as described in https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html and the resulting
                    adjacency matrix can then be passed as second argument to 'torch_geometric.nn.conv.GINConv.forward(x, adj_mat)'

        """
        # Get the number of vertices/nodes from the input vertex/node feature matrix 'x' that has shape (#vertices, #input_features).
        num_vertices = x.size(0)

        # Generate the TRANSPOSED sparse adjacency matrix from the edge indices
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html for details
        adj_mat_transposed = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_vertices, num_vertices))

        # Transpose this matrix to obtain the actual adjacency matrix
        adj_mat = adj_mat_transposed.t()

        # Do the GIN convolution passing the input vertex feature matrix and the sparse adjacency matrix
        x = self.gin_conv(x, adj_mat)

        # Apply layer normalization (this is the identity if 'layer_norm_mode' is None or 'None') and return the result
        return self.layer_norm(x, batch)

