# models/protein_models/utils.py

# Import custom modules
from . import model_definitions

# Define the function used to define models
def define_model(config_dict):
    # Try to get the object by name and return an error if it doesn't work
    try:
        model_obj = getattr(model_definitions, config_dict['name'])
    except AttributeError:
        err_msg = f"The protein model with name '{config_dict['name']}' is not defined in the file 'protein_models/model_definitions.py'."
        raise AttributeError(err_msg)

    # Instantiate the model object with the config dictionary
    return model_obj(config_dict)