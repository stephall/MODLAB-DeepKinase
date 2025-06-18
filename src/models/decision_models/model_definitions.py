#model_definitions.py

# Import public modules
import torch

# Import custom modules
from . import decision_layers
from .. import utils

# Define base decision model
class DecisionBaseModel(torch.nn.Module):
    def __init__(self, 
                 config_dict, 
                 x_combined_dim):
        """
        Args:
            config_dict (dict): Config dictionary.
            x_combined_dim (dict): Dimension of the combined latent features (molecules and proteins).
        """
        # Initialize the base class
        super().__init__()

        # Assign the inputs to class attributes
        self.config_dict    = config_dict
        self.x_combined_dim = x_combined_dim

    def forward(self, 
                x_combined):
        """ 
        Define the forward pass.

        Args:
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
        
        Return:
            (torch.tensor): Decision output.

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

class DecisionFCNNModel(DecisionBaseModel):
    def __init__(self, 
                 config_dict, 
                 x_combined_dim):
        """
        Args:
            config_dict (dict): Config dictionary.
            x_combined_dim (dict): Dimension of the combined latent features (molecules and proteins).
        """
        # Initialize the base/super/parent class
        super().__init__(config_dict, x_combined_dim)

        # Get the decision layer name
        decision_layer_name = self.config_dict['decision_layer_configs']['name']

        # Check that the relation representation type is one of the types required by the decision layer
        if self.config_dict['relation_representation_params']['type'] not in self.config_dict['decision_layer_configs']['required_relation_representation_types']:
            err_msg = f"The decision layer '{decision_layer_name}' requires the relation representation to be of type(s) '{list(self.config_dict['decision_layer_configs']['required_relation_representation_types'])}', " \
                      f"but got type '{self.config_dict['relation_representation_params']['type']}' instead."
            raise ValueError(err_msg)

        # Try to get a handle to the decision layer class using the decision layer name
        try:
            decision_layer_handle = getattr(decision_layers, decision_layer_name)
        except AttributeError:
            err_msg = f"The decision layer '{decision_layer_name}' is not defined in the file 'decision_layers.py'."
            raise AttributeError(err_msg)

        # Update the decision layer parameter dictionary for certain decision layers
        if decision_layer_name=='MeanPredictionLayer':
            # Assign the mean and standard deviation of the mean pX for each protein-molecule pair in the train set 
            # (that have been saved in 'relation_representation_params' of the decision model configuration) as 'output_shift'
            # and 'output_scale', respectively, decision layer parameters
            self.config_dict['decision_layer_configs']['output_shift'] = self.config_dict['relation_representation_params']['train_mean_pmp_pX_mean']
            self.config_dict['decision_layer_configs']['output_scale'] = self.config_dict['relation_representation_params']['train_std_pmp_pX_mean']
        elif decision_layer_name=='OrdinalDecisionLayer':
            # Use the number of activity levels as the number of ordinals in the decision layer
            self.config_dict['decision_layer_configs']['num_ordinals'] = self.config_dict['relation_representation_params']['num_activity_levels']
        elif decision_layer_name=='GridKernelInterpolationGPDecisionLayer':
            # Specify the likelihood based on the relation representation type
            if self.config_dict['relation_representation_params']['type']=='binary_labels':
                self.config_dict['decision_layer_configs']['likelihood'] = 'BernoulliLikelihood'
            else:
                err_msg = f"Likelihood for 'GridKernelInterpolationGPDecisionLayer' decision layer only specified for relation representation type 'binary_labels'."
                raise ValueError(err_msg)
        else:
            # For any other decision layers, do nothing
            pass

        # Initialize the decision layer object
        decision_layer_obj = decision_layer_handle(config_dict=self.config_dict['decision_layer_configs'])

        # Define a FCNN model
        # Remark: The output should be linear (i.e. not be non-linearly activated) thus pass non_linear_output=False
        self.fcnn = utils.define_fcnn('decision', 
                                      input_dim=self.x_combined_dim, 
                                      hidden_params=self.config_dict['hidden_params'], 
                                      output_dim=decision_layer_obj.input_dim,
                                      activation_fn=self.config_dict['activation_fn'],
                                      dropout=self.config_dict['dropout'],
                                      non_linear_output=False)

        # Assign the decision layer object to a class attribute
        # Remark: This is done here and not above because this way it will be shown
        #         after the FCNN (self.fcnn) part when 'printing' the decision model
        self.decision_layer = decision_layer_obj

    def display_model_information(self, 
                                  **kwargs):
        """
        Display model information.

        Remark: This method might for example be called when displaying metric values during training.

        """
        # Display the information of the decision layer
        self.decision_layer.display_model_information(**kwargs)

    def forward(self, 
                x_combined):
        """ 
        Define the forward pass.

        Args:
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
        
        Return:
            (torch.tensor): Decision output.

        """
        # Call the model up to (but not including) the decision layer
        x = self.forward_up_to_decision_layer(x_combined)

        # Pass this through the decision layer and return it
        return self.decision_layer(x)

    def forward_up_to_decision_layer(self, 
                                     x_combined):
        """ 
        Call the model up to (but not including) the decision layer. 
        
        Args:
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
        
        Return:
            (torch.tensor): Output of decision model before the its final 
                (i.e., decision) layer (i.e., what would be passed to the 
                decision layer).
        
        """
        # Pass combined latent features through the FCNN model and return the result.
        return self.fcnn(x_combined)

    def loss(self, 
             x_combined, 
             data):
        """ 
        Determine the loss. 
        
        Args:
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
            data (torch.data.Data): Data object.

        Return:
            (torch.tensor): Loss value.

        """
        # Call the model up to (but not including) the decision layer
        x = self.forward_up_to_decision_layer(x_combined)

        # Calculate the loss on the decision layer output (and the data used to construct x_combined)
        # and then return it
        return self.decision_layer.loss(x, data)

    def metric(self, 
               x_combined, 
               data):
        """ 
        Determine the metric. 
        
        Args:
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
            data (torch.data.Data): Data object.

        Return:
            (torch.tensor): Metric value.
        
        """
        # Call the model up to (but not including) the decision layer
        x = self.forward_up_to_decision_layer(x_combined)

        # Calculate the metric on the decision layer output (and the data used to construct x_combined)
        # and then return it
        return self.decision_layer.metric(x, data)

    def call_decision_layer_method(self, 
                                   method_name, 
                                   x_combined, 
                                   **kwarg):
        """ 
        Call a method of the decision layer. 

        Args:
            method_name (str): Name of the decision layer method.
            x_combined (torch.tensor): Combined latent features (molecules and proteins).
            **kwargs (dict): Forwarded to the decision layer method.

        Return:
            (torch.tensor): Output of the decision layer method.
        
        """
        # Try to get a handle on the decision layer method and if there is an attribute error, rethrow it
        try:
            method_handle = getattr(self.decision_layer, method_name)
        except AttributeError:
            err_msg = f"The decision layer of class '{type(self.decision_layer)}' has no method '{method_name}'."
            raise AttributeError(err_msg)

        # Call the model up to (but not including) the decision layer
        x = self.forward_up_to_decision_layer(x_combined)
    
        # Call the decision layer method using the model output up to the decision layer
        return method_handle(x, **kwarg)

    @property
    def output_dim(self):
        """ Return the output dimension of the model. """
        return self.config_dict['output_dim']

