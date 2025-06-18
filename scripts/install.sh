#!/bin/bash

################################################################################
### This script assumes that conda has been installed and aliased to 'conda',
### and a conda enviroment with python3.9 has been created and is activated.
################################################################################
# Inform user about installation
echo "Installing python modules in conda environment."

# Install pip using conda and upgrade it to the latest version
conda install -y pip
pip install --upgrade pip
  
# Install standard datascience modules
pip install numpy==1.23.4 matplotlib==3.6.1 pandas==1.5.1 scipy==1.9.3

# Install RDKit
pip install rdkit==2022.9.1

# Install ChEMBL structure pipeline (used to standardize and salt strip molecules) from https://github.com/chembl/ChEMBL_Structure_Pipeline
# Remark: This modules is only required to wash SMILES strings (when predicting for novel molecules)
pip install chembl_structure_pipeline==1.2.0

# Install (vanilla) pytorch
pip install torch==1.13.0

# Install pytorch geometric (and packages required for it) for torch=1.13.0 and CUDA 11.7 (cu117) using the wheels specified on 'https://data.pyg.org/whl/'
# pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric==2.3.1

# Install pyyaml (to read/write .yaml files) and hydra (config file management)
pip install pyyaml==6.0
pip install hydra-core==1.2.0

# Install ipykernel (for jupyter notebooks) and the ipywidgets (for additional functionality)
pip install ipykernel==6.20.2
pip install ipywidgets==8.0.2

# Additional auxiliary modules
pip install openpyxl==3.1.5

echo " "
echo "Installation done"
