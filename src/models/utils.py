# models/utils.py

# Import public modules
import collections
import torch
import numpy as np

def define_fcnn(name, 
                input_dim, 
                hidden_params, 
                output_dim, 
                activation_fn='ReLU', 
                dropout=None, 
                non_linear_output=True):
    """
    Define and return a Fully-Connected Neural Network (FCNN).

    Args:
        name (str): Name of the network that is used as prefix for all its parameters.
        input_dim (int): Input dimension.
        hidden_params (dict): Parameters used to construct the hidden layers.
        output_dim (int): Output dimension.
        activation_fn (str): Activation function to be used after each linear operation in each layer.
            (Default: 'ReLU')
        dropout (float or None): Dropout probability for each layer (applied after activation function).
            0 corresponds to no dropout and 1 would mean that everything is droped out.
            In case dropout is None, no dropout will be used.
            (Default: None)
        non_linear_output (bool): Should the output layer be activated (non-linear) or not?
            In case that the output should not be activated (non_linear_output=False), also no dropout
            will be applied to the output layer.
            (Default: True)

    Return:
        (torch.nn.Sequential): The FCNN as torch.nn.Sequential object. 
    
    """
    # Ensure that input_dim and output_dim are strictly positive integers
    # Input dim
    input_dim = int(input_dim)
    if input_dim<=0:
        err_msg = f"The passed input dimension must be a strictly positive number but was '{input_dim}' for FCNN '{name}'."
        raise ValueError(err_msg)
    # Output dim
    output_dim = int(output_dim)
    if output_dim<=0:
        err_msg = f"The passed output dimension must be a strictly positive number but was '{output_dim}' for FCNN '{name}'."
        raise ValueError(err_msg)

    # Check that 'non_linear_output' is a boolean
    if not isinstance(non_linear_output, bool):
        err_msg = f"The input 'non_linear_output' must be a boolean, got type '{type(non_linear_output)}' instead."
        raise TypeError(err_msg)

    # Use the content of hidden_params to determine the dimensions of the hidden layers 'hidden_dims'
    # corresponding to a list of integers in the form "[dim_1, dim_2, ...]"
    # Examples: 1) hidden_dims=[] <=> "No hidden layers"
    #           2) hidden_dims=[4] <=> "One hidden layer with dimension 4"
    #           3) hidden_dims=[10, 3] <=> "Two hidden layers of which the first has dimension 10 and the second has dimension 3"
    # Differ the two cases where the hidden_params dictionary contains the key 'hidden_dims' or not
    if 'hidden_dims' in hidden_params:
        # Use the value to the key 'hidden_dims' as the hidden_dims list
        hidden_dims = hidden_params['hidden_dims']
    else:
        # In case that 'hidden_dims' is not a key of the hidden_params dictionary,
        # 'num_hidden' must be a key and thus throw an error if it isn't
        if 'num_hidden' not in hidden_params:
            err_msg = f"The input 'hidden_params' passed for the definition of FCNN '{name}' must contain 'hidden_dims' or 'num_hidden' as key but does not contain either of them."
            raise ValueError(err_msg)

        # Interpolate the hidden dimensions ('hidden_dims') from the input to the output dimension
        # Cast num_hidden to an integer and check that it is positive
        num_hidden = int(hidden_params['num_hidden'])
        if num_hidden<0:
            err_msg = f"The input 'hidden_params' passed for the definition of FCNN '{name}' contains a negative value for 'num_hidden' which is not allowed."
            raise ValueError(err_msg)

        # Get the interpolation type from the hidden_params dictionary using 'exponential' as default
        interpolation_type = hidden_params.get('interpolation_type', 'exponential')

        # Differ cases for the interpolation type
        if interpolation_type=='linear':
            # Explanation: Use linear interpolation from input to the output dimension
            # Example: input_dim=16, output_dim=4 with num_hidden=1 would lead to hidden_dims=[10] which is 6 (additatively) away from both input_dim and output_dim

            # Construct an array of equally spaced (#hidden) steps between the input and output dimension
            # Remarks: 1) numpy.linspace(start, end, #points) constructs equally spaced (='linear') points between the start and the end
            #             where the start and end point are included in #points. Thus the number of interpolation steps is #points-2 and
            #             thus use 'num_hidden+2=#points' below. => Remove the start and endpoint by slicing '[1:-1]'
            #          2) The output of numpy.linspace() is an array containing float entries.
            float_hidden_dims = np.linspace(input_dim, output_dim, num_hidden+2)[1:-1]

            # Transform this array to a list where the elements are all rounded to integers
            hidden_dims = [int( np.round(float_hidden_dim) ) for float_hidden_dim in float_hidden_dims]
        elif interpolation_type=='exponential':
            # Explanation: Use exponential interpolation from input to the output dimension
            # Example: input_dim=16=2^4, output_dim=4=2^2 with num_hidden=1 would lead to hidden_dims=[8]=[2^3] which is 2^1=2 (multiplicatively) away from both input_dim and output_dim

            # Construct an array of equally spaced (#hidden) steps between the exponentials of the input and output dimension
            # Remarks: 1) numpy.linspace(start, end, #points) constructs equally spaced (='linear') points between the start and the end
            #             where the start and end point are included in #points. Thus the number of interpolation steps is #points-2 and
            #             thus use 'num_hidden+2=#points' below. => Remove the start and endpoint by slicing '[1:-1]'
            #          2) The output of numpy.linspace() is an array containing float entries.
            float_hidden_dims_exponentials = np.linspace(np.log(input_dim), np.log(output_dim), num_hidden+2)[1:-1]

            # Transform this array to a list where the base-2 exponentials are mapped to numbers that are all rounded to integers
            hidden_dims = [int( np.round( np.exp(float_hidden_dim_exponential) ) ) for float_hidden_dim_exponential in float_hidden_dims_exponentials]
        else:
            err_msg = f"The 'interpolation_type', used to determine the dimensions of the hidden layers of FCNN '{name}' and passed as key-value pair of 'hidden_params', was '{interpolation_type}' that is unexpected."
            raise ValueError(err_msg)

    # Try to get a handle to the activation function passed by its name (as string)
    try:
        activation_fn_handle = getattr(torch.nn, activation_fn)
    except AttributeError:
        err_msg = f"The activation function '{activation_fn}' for model part '{name}' is not defined in 'torch.nn'"
        raise AttributeError(err_msg)

    # Check that dropout is in [0, 1] if it is requested (not None)
    if dropout is not None:
        if not (0<=dropout and dropout<=1):
            err_msg = f"Dropout probability must be in [0, 1], got value {dropout} for model part '{name}' instead."
            raise ValueError(err_msg)

    # Create a sequential models object by constructing a list of layer models 
    # (actually tuples of layer names and layer model objects)
    seq_model_list = list()
    prev_layer_dim = input_dim
    layer_dims     = hidden_dims + [output_dim]
    for layer_index, layer_dim in enumerate(layer_dims):
        # Append a linear model
        seq_model_tuple = (f"{name}_Linear_{layer_index}", torch.nn.Linear(prev_layer_dim, layer_dim))
        seq_model_list.append( seq_model_tuple )

        # Append the activation function and dropout (if requested)
        # Remark: It might be that the last layer should not be non-linearly activated (non_linear_output=False).
        #         If this is the case, do not append the activation function for the last layer and also do not 
        #         use dropout for this layer.
        if layer_index<(len(layer_dims)-1) or non_linear_output:
            # Append the activation function
            seq_model_tuple = (f"{name}_{activation_fn}_{layer_index}", activation_fn_handle())
            seq_model_list.append(seq_model_tuple)

            # Append dropout if requested ('dropout' is not None)
            if dropout is not None:
                # Remark: Use passed value of 'dropout' for the dropout probability
                seq_model_tuple = (f"{name}_Dropout_{layer_index}", torch.nn.Dropout(p=dropout, inplace=False))
                seq_model_list.append(seq_model_tuple)

        # Assign the current layer dimension to the variable holding the previous layer's dimension
        prev_layer_dim = layer_dim

    # Pass the sequential model list as ordered dictionary argument to torch.nn.Sequential
    return torch.nn.Sequential( collections.OrderedDict(seq_model_list) )