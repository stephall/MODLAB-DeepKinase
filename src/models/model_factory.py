#model_factory.py

# Import public modules
import torch
import numpy as np

# Import custom modules
from .. import random_handling
from . import model_templates

# Define a function that will construct a model given an input configuration file
def define_model(config_dict, device=None):
    # In case that the device is not passed (is None), use CUDA if available and CPU otherwise
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a random handler and set a random seed
    random_handler = random_handling.RandomHandler()
    random_handler.set_seed(config_dict['model_init_seed'])

    # Get the model template name from the config dictionary using 'PairModel' if
    # it is not defined in the config dictionary
    model_template_name = config_dict.get('template', 'PairModel')

    # Try to get the model template (class) by name and return an error if it doesn't work.
    try:
        model_template_class  = getattr(model_templates, model_template_name)
    except AttributeError:
        err_msg = f"No model template with name '{model_template_name}' is defined in the file 'model_templates.py'."
        raise AttributeError(err_msg)
    
    # Initialize the model using the obtained model template (class)
    model = model_template_class(config_dict)

    # Move the model to the requested device
    model = model.to(device)

    # Put the model initially in evaluation mode
    model.eval()

    # Reset the random states
    random_handler.reset_states()

    return model
