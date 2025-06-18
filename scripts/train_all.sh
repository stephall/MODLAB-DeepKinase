#!/bin/bash

####################################################################################
### This script trains the models of the ensembles discussed in the article
####################################################################################
# 1) Train ensemble of optimal models (original)
LABEL='optimal_model_original'
python scripts/train.py label=$LABEL model.model_init_seed=1
python scripts/train.py label=$LABEL model.model_init_seed=2
python scripts/train.py label=$LABEL model.model_init_seed=3
python scripts/train.py label=$LABEL model.model_init_seed=4
python scripts/train.py label=$LABEL model.model_init_seed=5

# 2) Train ensemble of optimal models (where positive/negatives are scrambled)
LABEL='optimal_model_scrambled'
python scripts/train.py label=$LABEL model.model_init_seed=1 data_handling.data_preprocessing.train_data_scrambling.p_scrambling=1.0
python scripts/train.py label=$LABEL model.model_init_seed=2 data_handling.data_preprocessing.train_data_scrambling.p_scrambling=1.0
python scripts/train.py label=$LABEL model.model_init_seed=3 data_handling.data_preprocessing.train_data_scrambling.p_scrambling=1.0
python scripts/train.py label=$LABEL model.model_init_seed=4 data_handling.data_preprocessing.train_data_scrambling.p_scrambling=1.0
python scripts/train.py label=$LABEL model.model_init_seed=5 data_handling.data_preprocessing.train_data_scrambling.p_scrambling=1.0

# 3) Train ensemble of optimal models (where negatives are resampled)
LABEL='optimal_model_scrambled'
python scripts/train.py label=$LABEL model.model_init_seed=1 data_handling.data_preprocessing.non_measured_molecules_subsampling_params.segment_index=1
python scripts/train.py label=$LABEL model.model_init_seed=2 data_handling.data_preprocessing.non_measured_molecules_subsampling_params.segment_index=1
python scripts/train.py label=$LABEL model.model_init_seed=3 data_handling.data_preprocessing.non_measured_molecules_subsampling_params.segment_index=1
python scripts/train.py label=$LABEL model.model_init_seed=4 data_handling.data_preprocessing.non_measured_molecules_subsampling_params.segment_index=1
python scripts/train.py label=$LABEL model.model_init_seed=5 data_handling.data_preprocessing.non_measured_molecules_subsampling_params.segment_index=1

# 4) Train ensemble of baseline models
LABEL='baseline_model'
python scripts/train.py label=$LABEL model='random_forest_model' model.model_init_seed=1
python scripts/train.py label=$LABEL model='random_forest_model' model.model_init_seed=2
python scripts/train.py label=$LABEL model='random_forest_model' model.model_init_seed=3
python scripts/train.py label=$LABEL model='random_forest_model' model.model_init_seed=4
python scripts/train.py label=$LABEL model='random_forest_model' model.model_init_seed=5