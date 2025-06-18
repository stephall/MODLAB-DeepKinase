# models/combination_models/utils.py

# Import custom modules
from . import model_definitions

# Define the function used to define models
def define_model(config_dict, x_molecule_dim, x_protein_dim):
    # Try to get the object by name and return an error if it doesn't work.
    try:
        model_obj = getattr(model_definitions, config_dict['name'])
    except AttributeError:
        err_msg = f"The combination model with name '{config_dict['name']}' is not defined in the file 'combination_model/model_definitions.py'."
        raise AttributeError(err_msg)

    # Instantiate the model object with the config dictionary and the dimensions of the latent spaces
    return model_obj(config_dict, x_molecule_dim, x_protein_dim)