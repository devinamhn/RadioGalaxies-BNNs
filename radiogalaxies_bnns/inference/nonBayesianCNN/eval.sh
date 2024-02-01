#!/bin/bash

#SBATCH --job-name=cnneval
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/cnn/eval/out-slurm_%j.out


pwd;

nvidia-smi 

source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>eval"
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/nonBayesianCNN/cnn_eval.py
