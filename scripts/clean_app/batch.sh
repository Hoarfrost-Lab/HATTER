#!/bin/bash

#SBATCH --job-name=example-batch
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256gb
#SBATCH --time=7-00:00:00
#SBATCH --output=jobs/%x_%j.out
#SBATCH --error=jobs/%x_%j.err
#SBATCH --mail-user=Ashley.Babjac@uga.edu
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

ml Anaconda3/2024.02-1
ml CUDA/12.1.1

eval "$(conda shell.bash hook)"
conda activate fpa_env

python ./train-triplet.py --training_data split100 --model_name split100_triplet_layernorm_mc --epoch 7000
