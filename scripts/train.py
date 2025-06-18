# train.py

# Import public modules
import hydra
import omegaconf
import os
import sys
import yaml
from pathlib import Path

# Set the matplotlib config directory 
# Remark: The directory should be writtable and defined before 
#         importing matplotlib for the first time)
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/.matplotlib_configs/"

# This python script should be run from the main project directory of 
# which 'scripts' is a subdirectory containing this python script.
# If the main project directory (i.e. the current directory) is not in
# the path, add it to allow imports from all folders in the project.
current_dir_path = os.getcwd()
if current_dir_path not in sys.path:
    sys.path.append(current_dir_path)

# Import custom modules
from src import utils
from src import model_handler_factory
    
# Define global parameters related to the location of the to be used config file
config_dir_rel_path = '../configs' # Relative path to the folder containing the configuration files
config_name         = 'config.yaml' # Name of the to be used configuration file (located in the configuration file folder)

@hydra.main(config_path=config_dir_rel_path, config_name=config_name, version_base="1.1")
def main_exec(cfg: omegaconf.DictConfig):
    """ Define the main function that will be called when calling the python script as '__main__'."""
    # Display the current working directory
    print(f"The working/output directory of the run is: {os.getcwd()}")
    print()

    # Define the model handler (and get the updated config dictionary)
    model_handler, config_dict = model_handler_factory.define_model_handler(cfg)

    # Store the (updated) config dictionary as .yaml file
    file_path = r'./config.yaml'
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file)

    # Train the model for a certain number of epochs
    model_handler.train(config_dict['training']['num_epochs'])

    # Plot the learning curve and save it
    plot_save_path = str( Path(config_dict['training']['figures_dir_path'], 'learning_curve.png') )
    model_handler.plot_learning_curve(plot_save_path=plot_save_path)

# Check that the program is run as main
if __name__=="__main__":
    # Parse the base directory of the output files from the passed argument (sys.argv)
    output_base_dir = utils.parse_output_base_dir()
        
    # Get the arguments passed to the function
    # Remark: The first is the function name, so leave it out
    passed_args = sys.argv[1:]

    # Append the output files directory path to the system arguments 
    # in case it has not been passed in the arguments
    if not any(['hydra.run.dir' in passed_arg for passed_arg in passed_args]):
        # Construct the output file directory path for the current run
        run_dir_path = utils.construct_output_run_dir_path(passed_args, output_base_dir, config_dir_rel_path, config_name)

        # Append the this path to the system arguments
        sys.argv.append("hydra.run.dir=" + run_dir_path)

    # Call the main function (see definition above)
    main_exec()
else:
    err_msg = f"The program 'train.py' can only be run as main."
    raise OSError(err_msg)






