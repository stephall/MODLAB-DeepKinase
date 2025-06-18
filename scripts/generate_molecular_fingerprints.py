# generate_molecular_fingerprints.py

# Import public modules
import os
import pickle
import sys

# Add the parent directory (corresponding to the base project directory) 
# to the path to allow imports from all folders in the project
parent_dir_path = os.path.dirname(os.getcwd())
if parent_dir_path not in sys.path:
    sys.path.append(parent_dir_path)

# Import custom modules
from src import utils

# Check that the program is run as main
if __name__=="__main__":
    # Path to one of the trained models
    # Remark: It does not matter which one as only the config file is required
    #         to be able to construct the processed dataset and extract the
    #         non-stereochemical washed canonical SMILES (nswcs) strings from it.
    output_folder_path = '../trained_models/optimal_model_original/model_1'
    
    # Load all nswcs that were present in the the train, validation, or test set 
    # when training the model whose training outputs are in 'output_folder_path'.
    all_nswcs = utils.load_all_nswcs(output_folder_path)

    # Generate fingerprint mappers to be used to encode the molecules for the random forest model
    fp_nBits_list  = [4096]
    fp_radius_list = [3]
    for fp_nBits in fp_nBits_list:
        for fp_radius in fp_radius_list:
            fingerprint_mapper = utils.FingerprintMapper(all_nswcs, 
                                                         fp_radius=fp_radius, 
                                                         fp_nBits=fp_nBits)
        
            # Open a file and use dump() 
            file_path = f'../raw_data/fp_mappings/nswcs_to_{fingerprint_mapper.tag}_map.pickle'
            with open(file_path, 'wb') as file:  
                # A new file will be created 
                pickle.dump(fingerprint_mapper._smiles_to_fp_map, file) 
    
            print()
            print(f"Saved mapping in {file_path}")
            print()
        
            # Check that saving worked by loading the fingerprints and checking that they are the same
            with open(file_path, 'rb') as file:  
                # Call load method to deserialze 
                loaded = pickle.load(file)
        
            all_true = True
            for nswcs, features_loaded in loaded.items():
                features = fingerprint_mapper(nswcs)
                if features!=features_loaded:
                    all_true = False
            
            print(f"Does saving and loading produce the same fingerprints: {all_true}")
            print()
    
    print()
    print(f"All fingerprint mappings saved")
else:
    err_msg = f"The program 'train.py' can only be run as main."
    raise OSError(err_msg)






