#!/bin/bash

#SBATCH --job-name=cnn
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/cnn/datashuffle/out-slurm_%j.out


pwd;

nvidia-smi 
echo ">>>start task array $SLURM_ARRAY_TASK_ID"
echo "Running on:"
hostname
source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>training"
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/nonBayesianCNN/cnn_train.py $SLURM_ARRAY_TASK_ID

