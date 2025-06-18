# Import public modules
import os
import sys
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
from src import utils

# Ensure that the program is only run as main
if not __name__=='__main__':
    err_msg = f"The program 'generate_nswc_smiles_holdout.py' can only be run as main."
    raise SystemError(err_msg)

# Define global comment for article some of the compounds are from
article_url     = r'https://www.nature.com/articles/s41467-022-35692-6.epdf?sharing_token=hxJQKLa8zCIW-mkvPSxIq9RgN0jAjWel9jnR3ZoTv0MQemxG0KG5Gyx-8khDNH09wqaqe9XfCnbkLZFQLwQduNXme9qO6fLg9Vf8OMXyJpejOqtUpydlT1bYieDf8iYLobeTDMAL8xakC6YrCC_yuVbb36xTf0zC9-g11A7OS0c%3D'
article_name    = r'M. Moret \\textit{et al.}, Nature Communications, \\textbf{14}, 114 (2023).'
article_comment = 'Compound shown in article ' + r'\\href{' + article_url + '}{' + article_name + '}' + r'\\\\'

# Define a list for the commercially available compounds
# Remark: For this, define lists of dictionaries where each dictionary contains information about a molecule 
#         for which kinases activity should be predicted
commercial_molecules_list = [
    {
        'title': 'IPMM_1',
        'smiles': 'COC1=C(OC)C=CC(C2=NC3=CC=NN3C(C(F)F)=C2)=C1',
        'comment': article_comment,
        'measured_Kd_nM': [670, 620], # 640 reported
    },
    {
        'title': 'IPMM_2',
        'smiles': 'COC1=C(OC)C=CC(C2=NC3=C(C#N)C=NN3C(C(F)(F)F)=C2)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_3',
        'smiles': 'FC1=CC=C(C2=NOC(C3=CC(OC)=C(OC)C=C3)=N2)C=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_4',
        'smiles': 'COC1=C(OC)C=CC(C2=NC3=C(C#N)C=NN3C(C(F)F)=C2)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_5',
        'smiles': 'CC(C1=NN=C(C2=CC(OC)=C(OC)C=C2)S1)C',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_6',
        'smiles': 'COC1=C(OC)C=CC(C2=CN=C(N)C=C2C)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_7',
        'smiles': 'COC1=C(OC)C=CC(C2=CC(CCl)=NO2)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_8',
        'smiles': 'COC1=C(OC)C=CC(C(NC2=CC(C=CC=C3)=C3C=C2)=O)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_9',
        'smiles': 'OC1=C(OC)C=CC(C(N2)=NC(C3=CC=C(F)C=C3)=C2C4=CC=C(F)C=C4)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_10',
        'smiles': 'BrC(C=C1)=CC=C1C2=NC(C=CC=C3)=C3C(NC4=CC(OC)=C(OC)C=C4)=N2',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_11',
        'smiles': 'COC1=C(OC)C=CC(C2=NN=C(CCl)O2)=C1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_12',
        'smiles': 'NCC1=NOC(C2=CC(OC)=C(OC)C=C2)=N1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_13',
        'smiles': 'COC(C=C1C2=NOC(C3=CC=C(O)C=C3)=N2)=C(C=C1)OC',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_14',
        'smiles': 'ClCC1=C(C)OC(C2=CC(OC)=C(C=C2)OC)=N1',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_15',
        'smiles': 'COC(C=C1NC2=NC=CC=C2[N+]([O-])=O)=C(C=C1)OC',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
    {
        'title': 'IPMM_16',
        'smiles': 'COC(C=C1C2=C(C3=CC=C(OC)C(O)=C3)C=NO2)=C(C(OC)=C1)OC',
        'comment': article_comment,
        'measured_Kd_nM': None, # None reported
    },
]

# Define a list for the novel molecules
novel_molecules_list = [
    {
        'title': 'IPMM_17',
        'smiles': 'NC1=C2C(N(CCC3(COC3)C)N=C2C4=CC(O)=C(Cl)C=C4)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [64, 62], # 63 reported
    },
    {
        'title': 'IPMM_18',
        'smiles': 'NC1=C2C(N(CC3(COC3)C)N=C2C4=CC(O)=C(Cl)C=C4)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [46, 59], # 52 reported
    },
    {
        'title': 'IPMM_19',
        'smiles': 'NC1=C2C(N(CC3(COC3)C)N=C2C4=CC(OC)=C(Br)C=C4)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [170, 150], # 160 reported
    },
    {
        'title': 'IPMM_20',
        'smiles': 'NC1=C2C(N(C(C)C)N=C2C3=CC(C4CC4)=C(O)C(Cl)=C3)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [130, 110], # 120 reported
    },
    {
        'title': 'IPMM_21',
        'smiles': 'NC1=C2C(N(C(C)C)N=C2C3=CC(C4CC4)=C(O)C(C5CC5)=C3)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [380, 210], # 290 reported
    },
    {
        'title': 'IPMM_22',
        'smiles': 'NC1=C2C(N(C(C)C)N=C2C3=CC(Cl)=C(O)C(Cl)=C3)=NC=N1',
        'comment': article_comment,
        'measured_Kd_nM': [11, 15], # 13 reported
    },
    {
        'title': 'IPNS_A',
        'smiles': 'NC1=C2C(C(CCC3(C)COC3)N=C2C4=CC(O)=C(Br)C=C4)=NC=N1',
        'comment': r'Not synthesized yet, but could be synthesized.\\\\',
    },
    {
        'title': 'IPNS_B',
        'smiles': 'CC(C1CCN(C)C1)N2C3=NC=NC(N)=C3C(C4=CC(O)=C(Br)C=C4)=N2',
        'comment': r'Not synthesized yet, but could be synthesized.\\\\',
    }
]
# Add the lists to obtain a list of molecule dictionaries containing the 'IPMM' molcules
ipmm_molecules_list = commercial_molecules_list + novel_molecules_list

# Generate a list of non-stereochemical washed canonical SMILES (nswcs) strings of these molecules
ipmm_molecules_nswcs = list()
for molecule_dict in ipmm_molecules_list:
    # Transform the molecule SMILES string to a non-stereochemical washed canonical SMILES (nswcs) string
    # and append it to the corresponding list
    molecule_nswcs = utils.get_nswc_smiles(molecule_dict['smiles'])
    ipmm_molecules_nswcs.append(molecule_nswcs)


# Ensure that the raw data folder exists and throw an error otherwise
raw_data_base_dir = './raw_data'
if not os.path.isdir(raw_data_base_dir):
    err_msg = f"No raw-data directory found in: {raw_data_base_dir}"
    raise FileNotFoundError(err_msg)

# Save the list as a .tsv file and inform the user
file_path = Path(raw_data_base_dir, 'nswc_smiles_holdout.tsv')
save_df = pd.DataFrame({'non_stereochemical_washed_canonical_smiles': ipmm_molecules_nswcs})
save_df.to_csv(file_path, index=False)
print(f"Saved the list of holdout molecules in: {file_path}")