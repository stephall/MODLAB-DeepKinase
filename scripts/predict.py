# predict.py

# Import public modules
import argparse
import collections
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# This python script should be run from the main project directory of 
# which 'scripts' is a subdirectory containing this python script.
# If the main project directory (i.e. the current directory) is not in
# the path, add it to allow imports from all folders in the project.
current_dir_path = os.getcwd()
if current_dir_path not in sys.path:
    sys.path.append(current_dir_path)

# Import custom modules
from src import model_handler_factory
from src import prediction
from src import smiles_to_graph_mapping
from src import utils

# Make sure that this function can only be run as main
if __name__!='__main__':
    err_msg = f"The script 'predict.py' can only be run as main."
    raise OSError(err_msg)

# Define the argument parser
parser = argparse.ArgumentParser(
                    prog = 'predict',
                    description = 'Predict for molecules defined in input file.')

# Define the arguments
parser.add_argument('input', help='Specify the path to the input file containing the molecules. This file should be a single-column .tsv file containing SMILES strings of the molecules for which we want to predict.')
parser.add_argument('--output', type=str, default='./prediction/predictions.tsv', help='Specify the path of the output file containing the predictions.')
parser.add_argument('--model', type=str, default='optimal_model_original', help="Specify the path to a model (ensemble) directory. (Default: 'optimal_model_original')")
parser.add_argument('--batch_size', type=int, default=128, help='Specify the batch size that should be used when predicting.')

# Parse the arguments
args = parser.parse_args()

# Assign the parsed arguments to variables
input_file_path  = args.input
output_file_path = args.output
ensemble_name    = args.model
batch_size       = args.batch_size

# Predictions with the baseline model is not implemented
if ensemble_name=='baseline_model':
    raise NotImplementedError(f"Predictions are not implemented for the baseline model.")

###################################################################################################
### Part 1: Parse and preprocess the input of molecular SMILES strings
###################################################################################################
# Load the input 
if not os.path.isfile(input_file_path):
    err_msg = f"No input file found in the path: {input_file_path}"
    raise FileNotFoundError(err_msg)

input_df = pd.read_csv(input_file_path, sep='\t', header=None)
print(f"Loaded inputs from: {input_file_path}")
input_df.rename(columns = {0: 'smiles'}, inplace=True)
smiles_index_map = {smiles: index for index, smiles in enumerate(input_df['smiles'])}

# Try to map all SMILES strings to non-stereochemical washed canonical SMILES (nswcs) strings
nswcs_to_smiles_list_map = collections.defaultdict(list)
nswcs_list = list()
non_nswcs_mappable_smiles_list = list()
for smiles in input_df['smiles']:
    try:
        nswcs = utils.get_nswc_smiles(smiles)
        nswcs_to_smiles_list_map[nswcs].append(smiles)
    except:
        nswcs = None
        non_nswcs_mappable_smiles_list.append(smiles)
    
    nswcs_list.append(nswcs)

input_df['nswcs'] = nswcs_list

# Inform user if certain SMILES strings could not be mapped to nswcs
if 0<len(non_nswcs_mappable_smiles_list):
    stringified_smiles_list = '\n'.join([f" '{smiles}'" for smiles in non_nswcs_mappable_smiles_list])
    print(f"The following SMILES strings could not be mapped to a non-stereochemical washed canonical SMILES (nswcs) string:\n{stringified_smiles_list}\nNo predictions will be made for these SMILES strings!\n")

# Generate a map from each nswcs to its corresponding molecular graph
smiles_to_graph_mapper = smiles_to_graph_mapping.SmilesToGraphMapper()
nswcs_list = list(nswcs_to_smiles_list_map.keys())
non_graph_mappable_nswcs_list = list()
nswcs_to_graph_map = dict()
for nswcs in nswcs_list:
    try:
        nswcs_to_graph_map[nswcs] = smiles_to_graph_mapper(nswcs)
    except smiles_to_graph_mapping.MoleculeCannotBeMappedToGraph:
        non_graph_mappable_nswcs_list.append(nswcs)

# Inform user if the nswcs of certain SMILES strings could not be mapped to a graph
if 0<len(non_graph_mappable_nswcs_list):
    non_graph_mappable_smiles_list = list()
    for nswcs in non_graph_mappable_nswcs_list:
        non_graph_mappable_smiles_list += nswcs_to_smiles_list_map[nswcs]
        
    stringified_smiles_list = '\n'.join([f" '{smiles}'" for smiles in non_graph_mappable_smiles_list])
    print(f"The the following SMILES strings (i.e., their non-stereochemical washed canonical SMILES strings) could not be mapped to a molecular graph in the form expected by the prediction models:\n{stringified_smiles_list}\nFor example, a SMILES string might contain an atom that is not allowed for the molecular-graph input for the prediction model.\nNo predictions will be made for these SMILES strings!\n")

# Check that the smiles to graph mapper is not empty and throw an error if it is
if len(nswcs_to_graph_map)==0:
    err_msg = f"None of the passed SMILES strings can be used as model input.\nThe input smiles might not be mappable to non-stereochemical washed canonical SMILES (nswcs) strings or (if they are mappable) their nswcs could not be mapped to the molecular-graph input for the prediction model."
    raise ValueError(err_msg)

###################################################################################################
### Part 2: Predict for each model
###################################################################################################
# Define the model ensemble's directory
ensemble_dir_path = str(Path('./trained_models', ensemble_name))

# Determine all the model names in the ensemble directory
model_names = list()
for file_name in os.listdir(ensemble_dir_path):
    # Skip .tar.gz files or '.ipynb_checkpoints'
    if file_name.endswith('.tar.gz') or file_name=='.ipynb_checkpoints':
        continue
    model_names.append(file_name)

model_names.sort()

print(f"Predicting for {len(model_names)} ensemble models...")
# Loop over the models and predict for each model
for model_index, model_name in enumerate(model_names):    
    # Construct path to the file (that should correspond to the output folder of one of the ensemble's models)
    file_path = str(Path(ensemble_dir_path, model_name))

    # Define the model handler
    model_handler, config_dict = model_handler_factory.define_model_handler(file_path, 
                                                                            load_data=False,
                                                                            temp_dir_path='./prediction/temp',
                                                                            silent_mode=True)
    
    # Load the model
    model_handler.load()

    # Initialize a molecules predictor object for the current model
    predictor = prediction.Predictor(model_handler.model, 
                                     config_dict['model'],
                                     nswcs_to_graph_map, 
                                     batch_size=batch_size)

    # Predict and rename the prediction column
    print(f"Predicting for  {model_name}...")
    model_prediction_df = predictor.run()

    # Rename the prediction column to '<model_name>_prediction'
    model_prediction_df.rename(columns={'prediction': f'{model_name}_prediction'}, inplace=True)

    # If there are SMILES that cannot be mapped to an nswcs, we want to add None to all other entries for 
    # the predictions. If this is the case, add an additional row with None in all columns to achieve this.
    if 0<len(non_nswcs_mappable_smiles_list):
        protein_ids = list(set(list(model_prediction_df['protein_id'])))
        additional_row_dict = dict()
        for column in model_prediction_df:
            if column=='protein_id':
                column_vals = protein_ids
            elif column=='nswcs':
                column_vals = [None]*len(protein_ids)
            else:
                column_vals = [np.nan]*len(protein_ids)
            additional_row_dict[column] = column_vals
        additional_row_df = pd.DataFrame(additional_row_dict)
        model_prediction_df = pd.concat([model_prediction_df, additional_row_df], ignore_index=True)

    # If there are nswcs of SMILES that cannot be mapped to a graph, we want to add None to each of the
    # nswcs that cannot be mapped to a graph. If this is the case, add additional rows with None in all 
    # columns (but 'nswcs') to achieve this.
    if 0<len(non_graph_mappable_nswcs_list):
        additional_rows_dict = collections.defaultdict(list)
        protein_ids = list(set(list(model_prediction_df['protein_id'])))
        for nswcs in non_graph_mappable_nswcs_list:
            for column in model_prediction_df:
                if column=='protein_id':
                    column_vals = protein_ids
                elif column=='nswcs':
                    column_vals = [nswcs]*len(protein_ids)
                else:
                    column_vals = [np.nan]*len(protein_ids)
                additional_rows_dict[column] += column_vals
        
        additional_rows_df = pd.DataFrame(additional_rows_dict)
        model_prediction_df = pd.concat([model_prediction_df, additional_rows_df], ignore_index=True)
    
    # Merge the model's predictions with all the other predictions
    # Differ the case where the current model is the first model or not
    if model_index==0:
        # Initialize the global prediction DataFrame using outer-join between
        # the input DataFrame and the current model's prediction DataFrame
        prediction_df = input_df.merge(model_prediction_df, 
                                       how='outer', 
                                       on='nswcs')
    else:
        # Update the global prediction DataFrame by inner-join it with
        # the current model's prediction DataFrame
        prediction_df = prediction_df.merge(model_prediction_df, 
                                            how='inner', 
                                            on=['nswcs', 'protein_id'])

print('Predictions done.')

###################################################################################################
### Part 3: Save predictions
###################################################################################################
# Determine the ensemble prediction as the mean over all models for each molecule-protein pair (i.e., row)
prediction_columns = [column for column in prediction_df.columns if column.endswith('prediction')]
prediction_df['ensemble_prediction'] = prediction_df[prediction_columns].mean(axis=1)

# Add the UniProt name of the proteins to each prediction
protein_name = 'Protein UniProt Name'
protein_id_name_df = pd.read_csv('./raw_data/protein_chembl_ids_and_uniprot_names.tsv', 
                                 sep='\t')

protein_id_uniprot_entry_name_map = dict()
for protein_id, uniprot_entry_name in zip(protein_id_name_df['protein_id'], protein_id_name_df['Entry Name (UniProt)']):
    protein_id_uniprot_entry_name_map[protein_id] = uniprot_entry_name
prediction_df[protein_name] = prediction_df['protein_id'].apply(lambda x: protein_id_uniprot_entry_name_map[x] if x is not None else None)

# Make the added protein name column as the second column
columns = list(prediction_df.columns)
columns.remove(protein_name)
columns.insert(1, protein_name)
prediction_df = prediction_df[columns]

# Sort the predictions DataFrame based on molecules and proteins
prediction_df['input_molecule_index'] = prediction_df['smiles'].apply(lambda x: smiles_index_map[x])
prediction_df.sort_values(by=['input_molecule_index', protein_name], inplace=True)
prediction_df.reset_index(drop=True, inplace=True)

# Rename the 'smiles' column
prediction_df.rename(columns={'smiles': 'Molecule SMILES string'}, inplace=True)

# Drop the auxiliary columns 'input_molecule_index', 'nswcs', and 'protein_id'
prediction_df.drop(columns=['input_molecule_index', 'nswcs', 'protein_id'], inplace=True)

# Save the predictions DataFrame
if output_file_path.endswith('.tsv'):
    sep='\t'
else:
    sep=','

print(f"Saving predictions to file: {output_file_path}")
prediction_df.to_csv(output_file_path, 
                     sep=sep,
                     index=False)
print('Saving done.')



