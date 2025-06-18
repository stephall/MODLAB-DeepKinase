#!/bin/bash

################################################################################
### This script unpacks various tarred files.
################################################################################
echo 'Start unpacking...'
echo ''
# Untar the Activities_Kinases.tsv in <root>/raw_data
cd raw_data
tar -xzvf Activities_Kinases.tsv.tar.gz

# Untar folders of trained models in <root>/trained_models/
cd ../trained_models/optimal_model_original
for ((i=1;i<=5;i++)); do
    tar -xzvf model_${i}.tar.gz
done
cd ../optimal_model_scrambled
for ((i=1;i<=5;i++)); do
    tar -xzvf model_${i}.tar.gz
done
cd ../optimal_model_resampled
for ((i=1;i<=5;i++)); do
    tar -xzvf model_${i}.tar.gz
done
cd ../baseline_model
for ((i=1;i<=5;i++)); do
    tar -xzvf model_${i}.tar.gz
done

# Untar the UniProt-Human-List file in <root>/dataset_construction/tables/input/
cd ../../dataset_construction/tables/input
tar -xzvf UniProt_Human_list.tsv.tar.gz

echo ''
echo 'Unpacking done.'