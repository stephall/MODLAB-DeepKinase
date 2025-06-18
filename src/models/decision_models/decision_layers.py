#decision_layers.py

# Import public modules
import torch

# Define the base class for the decision layer, which is the last layer that maps to the decision space
# and thus depends on the decision space (i.e. the representation of the activities).
class BaseDecisionLayer(torch.nn.Module):
    def __init__(self, 
                 config_dict={}):
        """
        Args:
            config_dict (dict): Config dictionary.
                (Default: {})
        """
        # Initialize the base class
        super().__init__()

        # Assign config_dict to a class attribute of the same name
        self.config_dict = config_dict

    @property
    def input_dim(self):
        """ Return the input dimension of the layer. """
        raise NotImplementedError("Any child class must implement the (property) method 'input_dim'.")

    @property
    def output_dim(self):
        """ Return the output dimension of the layer. """
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

    def forward(self, 
                x):
        """ 
        Call the layer.

        Args:
            x (torch.tensor): Input tensor.
        
        Return:
            (torch.tensor): Output tensor.

        """
        raise NotImplementedError("Any child class must implement the method 'forward'.")

    def loss(self, 
             x, 
             data):
        """ 
        Calculate the loss for the input tensor and the data that was used to produce
        the input tensor.

        Args:
            x (torch.tensor): Input tensor.
            data (torch.data): Data object to which the input tensor corresponds to 
                and that contains the target variables required for the loss.
        
        Return:
            (torch.tensor): Loss object.

        """
        raise NotImplementedError("Any child class must implement the (property) method 'loss'.")

    def metric(self, 
               x, 
               data):
        """ 
        Calculate the metric for the input tensor and the data that was used to produce
        the input tensor.

        Args:
            x (torch.tensor): Input tensor.
            data (torch.data): Data object to which the input tensor corresponds to 
                and that contains the target variables required for the loss.
        
        Return:
            (torch.tensor): Metric object.

        """
        raise NotImplementedError("Any child class must implement the (property) method 'metric'.")

# Define different decision layers
# 1) Define an identity decision layer that maps a input to itself
class IdentityDecisionLayer(BaseDecisionLayer):
    def __init__(self, 
                 input_dim=1, 
                 config_dict={}):
        """
        Args:
            input_dim (int): Dimension of the input.
            config_dict (dict): Config dictionary.
                (Default: {})
        """
        # Initialize the base class
        super().__init__(config_dict=config_dict)

        # Assign the method inputs to class attributes
        self._input_dim = input_dim

        # This layer corresponds to an identity later
        self.id_layer = torch.nn.Identity()

    @property
    def input_dim(self):
        """ Return the input dimension of the layer. """
        return self._input_dim

    @property
    def output_dim(self):
        """ Return the output dimension of the layer. """
        # As the layer maps an input to itself, the output dimension is equal to the input dimension
        return self._input_dim

    def forward(self, 
                x):
        """ 
        Define the forward pass.

        Args:
            x (torch.tensor): Input tensor
        
        Return:
            (torch.tensor): Output tensor.

        """
        return self.id_layer(x).squeeze()


# 2) Define a binary classification decision layer based on the identity decision layer
class BinaryClassificationDecisionLayer(IdentityDecisionLayer):
    def __init__(self, 
                 config_dict={}):
        """
        Args:
            config_dict (dict): Config dictionary.
                (Default: {})
        """
        # Initialize the base class
        # Remark: For binary decision, a logit is predicted before the decision
        #         layer and thus the input dimension of the decision layer is 1.
        super().__init__(input_dim=1, config_dict=config_dict)

        # Define the loss and the metric
        self.loss_fn   = lambda model_logit, data_label: torch.nn.functional.binary_cross_entropy_with_logits(model_logit, data_label.float(), reduction='sum')
        self.metric_fn = lambda model_logit, data_label: torch.nn.functional.binary_cross_entropy_with_logits(model_logit, data_label.float(), reduction='sum')

    def get_logit(self, 
                  x):
        """
        Determine the logit of the input and return it.

        Args:
            x (torch.tensor): Input tensor.
        
        Return:
            (torch.tensor): Tensor containing the logit of the input

        """
        # Call the forward method of the base/super/parent class
        return super().forward(x)

    def get_label_1_prob(self, 
                         x):
        """
        Determine the label-1 probability of the input and return it.

        Args:
            x (torch.tensor): Input tensor.
        
        Return:
            (torch.tensor): Tensor containing the label-1 probability for the input.

        """
        # Get the logit values for the input
        logit = self.get_logit(x)

        # Use a sigmoid to map the logit values to the probabilities of label-1
        return torch.sigmoid(logit)

    def forward(self, 
                x):
        """ 
        Define the forward pass.

        Args:
            x (torch.tensor): Input tensor
        
        Return:
            (torch.tensor): Output tensor.

        """
        # Return the label-1 probability of the input
        return self.get_label_1_prob(x)

    def get_label(self, 
                  x, 
                  decision_threshold=0.5):
        """
        Determine the label of the input and return it.

        Args:
            x (torch.tensor): Input tensor.
            decision_threshold (float): Decision threshold.
                (Default: 0.5)
        
        Return:
            (torch.tensor): Tensor containing the label (0 or 1) for the input.

        """
        # Get the label-1 probability for the input
        label_1_prob = self.get_label_1_prob(x)

        # Initialize the label tensor to a zeros tensor of the same shape as label_1_prob
        label = torch.zeros_like(label_1_prob)

        # Assign 1 to all labels that have a label-1 probability bigger than the decision threshold
        ix = torch.where(decision_threshold<=label_1_prob)
        label[ix] = 1

        # Return the label
        return label

    def loss(self, 
             x, 
             data):
        """ 
        Calculate the loss for the input tensor and the data that was used to produce
        the input tensor.

        Args:
            x (torch.tensor): Input tensor.
            data (torch.data): Data object to which the input tensor corresponds to 
                and that contains the target variables required for the loss.
        
        Return:
            (torch.tensor): Loss object.

        """
        # Get the logits for the input tensor
        model_relation_logit = self.get_logit(x)

        # Get the data relation label
        data_relation_label = data.y

        # Calculate the loss
        return self.loss_fn(model_relation_logit, data_relation_label)

    def metric(self, 
               x, 
               data):
        """ 
        Calculate the metric for the input tensor and the data that was used to produce
        the input tensor.

        Args:
            x (torch.tensor): Input tensor.
            data (torch.data): Data object to which the input tensor corresponds to 
                and that contains the target variables required for the loss.
        
        Return:
            (torch.tensor): Metric object.

        """        
        ##########################################################################
        # Metric is calculated on model-logits and data-labels 
        ##########################################################################
        # Get the logits for the input tensor
        model_relation_logit = self.get_logit(x)

        # Get the data relation label
        data_relation_label = data.y

        # Calculate the metric
        return self.metric_fn(model_relation_logit, data_relation_label)


