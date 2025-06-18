#!/bin/bash
################################################################################
### This script generates the activities datasets and other auxiliary files.
################################################################################

# Generate tables containing the protein information (UniProt) for the Kinases 
# and the activity values (ChEMBL) of molecule-protein pairs.

# This bash script will create the tables:
# 1) './tables/output/proteins/Kinases_table.tsv'                  (Protein information of the Kinases)
# 2) './tables/output/activities/Activities_Kinases.tsv'           (Activities of molecule-Kinases pairs)
# 3) './tables/output/molecules/nswc_smiles_kinases_chembl.tsv'    (Non-stereochemical washed canonical SMILES strings of ALL molecules measured for ANY kinase)
# 4) './tables/output/molecules/molecules_from_qmugs_summary.tsv'  (Non-stereochemical washed canonical SMILES strings and molecular weight of ALL molecules in QMugs summary)

echo $1

######################################################################################################################
# Create the Protein Information Tables for the Kinases using UniProt Data
######################################################################################################################
# Generate a table containing the UniProt information for each kinase
# Remark: This function will create the table: './tables/output/proteins/Kinases_table.tsv'
python src/generate_kinases_table.py
echo " "

######################################################################################################################
# Create the Activity Tables for the Kinases using ChEMBL Data
######################################################################################################################
# Extract the activities from ChEMBL (in case they were not already downloaded)
# Remark: This function will create the folder: './ChEMBL_Data' (and its subfolders).
#python src/extract_activities_from_chembl.py

# Combine the ChEMBL data containing molecule-protein binding (MPB) measurements
# for the Kinases thereby creating a MPB table for both.
# Remark: This function will create the table:
#         './tables/output/activities/Activities_Kinases.tsv'
python src/combine_chembl_data.py
echo " "

######################################################################################################################
# Create a list containing the non-stereochemical washed canonical (nswc) SMILES strings of ALL molecules that have
# been measured for ANY kinase on ChEMBL
######################################################################################################################
# Generate this list
# Remark: The following function will create the file: './tables/outputs/molecules/nswc_smiles_kinases_chembl.tsv'
python src/generate_nswc_smiles_kinases_chembl.py
echo " "

######################################################################################################################
# Create a table containing the non-stereochemical washed canonical (nswc) SMILES strings and molecular weights of 
# ALL molecules found in the QMugs summary
######################################################################################################################
# Generate this table
# Remark: The following function will create the table: './tables/outputs/molecules/molecules_from_qmugs_summary.tsv'
python src/generate_molecules_from_qmugs_summary.py
echo " "

echo "Dataset generation finished"