# Repository containing the code of LINK_TO_PUBLISHED_ARTICLE.

## Folder structure
We will denote the root-level of the project directory as *PROJECT*.
</br>
The folder structure of the project directory is
</br>
*PROJECT*:
- *configs*: Contains all the config files for model training.
- *dataset_construction*: Contains all files related to the construction of the dataset.
- *notebooks*:
    - *saved*: Contains files of saved notebook outputs.
    - *HyperparameterAnalysis.ipynb*: Notebook used to visualize the results of the hyperparameter analysis.
    - *TestSetEvaluation.ipynb*: Notebook used to create the article figure.
- *outputs*: Will contain outputs of model training.
- *raw_data*: Contains the 'raw' (*i.e.*, unprocessed) data files (of the data set) that will be loaded and further preprocessed
- *scripts*: Contains scripts used for the Setup or Model training (discussed below).
- *src*: Contains all source files.
- *trained_models*: Contains files related to the trained models used in the article.
- *LICENSE*: The license file.
- *README.md*: This file.

## Setup
Create a Conda (python3.9) environment (with a name set to $ENV_NAME):
```bash
conda create -n $ENV_NAME python=3.9
```

Activate this Conda environment:
```bash
conda activate $ENV_NAME
```
In the following, we will indicate script execution from the root-level of the project directory and within the Conda environment by adding \[PROJECT\] and (conda-env), respectively, to the prompt.

Install all required python modules:
```bash
[PROJECT] (conda-env) bash scripts/install.sh
```

Unpack various *.tar.gz files:
```bash
[PROJECT] (conda-env) bash scripts/unpack.sh
```

Create the molecular fingerprings (required for the baseline model) by running the following script within *PROJECT/scripts*:
```bash
[PROJECT/scripts] (conda-env) python generate_molecular_fingerprints.py
```

## Model training
### (2a) Train optimal (default) model
```bash
[PROJECT] (conda-env) python scripts/train.py
```
Remarks:
- The results from training (e.g., model checkpoints or log files) will be saved in *PROJECT/outputs/\<date-now\>/\<model-name\>/\<label\>*
where \<date-now\> is the date of execution in the format 'YYYY-MM-DD', \<model-name\> is the model name as specified in the chosen model file located in *PROJECT/configs/model*, and \<label\> is a custom label (label='default' if not specified).
- The first time this script is executed, the file *raw_data/smiles_to_graph_map.pickle* will be created (and saved) during preprocessing.
This file will be loaded (and not re-created) in following script executions.

### (2b) Pass in some configurations
For example the optimal (default) model can be trained while seting a custom label ('test') for the model output files:
```bash
[PROJECT] (conda-env) python scripts/train.py label='test'
```
Remarks:
- Different configurations can be separated with spaces.
- The configurations have a tree structure that can be representated by dictionaries of dictionaries with multiple levels. For example configs['training']['num_epochs'] corresponds to the number of epochs used for training.
Sub-configurations are indicated by '.', e.g. config['a']['b']=x could be set by passing in 'a.b=x'.

The following script will train the model for 5 epochs and use a custom label for the output files:
```bash
[PROJECT] (conda-env) python scripts/train.py label='test' training.num_epochs=5
```

The following script will train the baseline model, which is a random forest model:
```bash
[PROJECT] (conda-env) python scripts/train.py label='baseline_model' model='random_forest_model'
```

### (2c) Sequentially train all the optimal models (and the baseline model) discussed in the article
The following script will sequentially train all 15 models (3 ensembles with 5 models each) discussed in the article.
```bash
[PROJECT] (conda-env) bash scripts/train_all.sh
```

## Predictions
### (2a) Predict for the optimal (default) model
```bash
[PROJECT] (conda-env) python scripts/predict.py INPUT_FILE_PATH
```
where *INPUT_FILE_PATH* is the path to a .tsv file containing a single column with SMILES strings (without a column header).
The file *PROJECT/prediction/example_input.tsv* represents an example for such an input file.
Prediction per ensemble model will be stored as a table in the file *PROJECT/prediction/predictions.tsv* (by default).
In case that a SMILES string can not be mapped to non-stereochemical washed canonical SMILES (nswcs) strings or (if they can be mapped to nswcs) its nswcs cannot be mapped to a molecular graph in the form expected as input for the prediction models, no predictions will be made for this SMILES string.

### (2b) Specify the output file name
```bash
[PROJECT] (conda-env) python scripts/predict.py INPUT_FILE_PATH --output OUTPUT_FILE_PATH
```
where *OUTPUT_FILE_PATH* is the path of the file in which the predictions will be stored in.

### (2c) Specify the model for predictions
```bash
[PROJECT] (conda-env) python scripts/predict.py INPUT_FILE_PATH --model <model>
```
where \<model\> is the name of the model ensemble (in *PROJECT/trained_models*) to be used for prediction.
Remark: All but the baseline model can be selected for predictions.

## Re-constructing the dataset from scratch
Download the file *summary.csv* from the [data-repository](https://doi.org/10.3929/ethz-b-000482129) of [QMugs](https://www.nature.com/articles/s41597-022-01390-7) and save it as *qmugs_summary.csv* in PROJECT/dataset_construction/tables/input.
</br>
The other files already in *PROJECT/dataset_construction/tables/input* are:
- *KinHub_List.xlsx* has been extracted from [KinHub](http://www.kinhub.org/index.html).
- Both *Keyword-Kinases(KW-0418)_UniProt.tsv* and *UniProt_Human_list.tsv* are each results of [UniProt](https://www.uniprot.org/) queries.

Note that PROJECT/dataset_construction contains a folder ChEMBL_Data that itself contains bioactivity data that has been extracted for each of the kinases from [ChEMBL](https://www.ebi.ac.uk/chembl/).

To re-construct the dataset, execute the following script within the dataset_construction folder (*PROJECT/dataset_construction*) and inside the Conda environemt:
```bash
[PROJECT/dataset_construction] (conda-env) bash generate_dataset.sh
```
This script will create the following files in the folder *PROJECT/dataset_construction/tables/output*:
1. *activities/Activities_Kinases.tsv*
2. *molecules/molecules_from_qmugs_summary.tsv*
3. *molecules/nswc_smiles_kinases_chembl.tsv*
4. *proteins/Kinases_table.tsv*

The first three files (1-3) can be copied to *PROJECT/raw_data*.
</br>
Note that the order of the rows of the newly created *Activities_Kinases.tsv* might differ from the already existing file in *PROJECT/raw_data*.
This is fine, as the data is sorted during data preprocessing before training a model.

## Funding
This project was launched at ETH Zurich in May 2022 and completed in February 2025.
It was supported by the Swiss National Science Foundation (grant numbers 205321_182176 and 1-007655-000).