#!/bin/bash
# You need to modify this path
DATASET_DIR="./demos-lifelong-transfer"

# You need to modify this path as your workspace
WORKSPACE="./demos-lifelong-transfer/pub_demos_cnn"

DEV_SUBTASK_A_DIR="demos_data"

BACKEND="pytorch"
HOLDOUT_FOLD=1
GPU_ID=0
FEATURE="logmel"
CONVLAYER=1 


############ Development subtask A ############
# Train model for subtask A
python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --validation --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask A
python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --validation --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda



############ Full train subtask A ############
# Train on full development data
python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --cuda
	
# Inference evaluation data
python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda


#done
#done
