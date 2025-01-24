#!/bin/bash

#SBATCH --job-name=rgzeval
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/rgz/out-slurm_%j.out


pwd;

nvidia-smi 

source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>eval"
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/eval/rgz/rgz_inference/rgz_outliers.py #mightee_inference.py
