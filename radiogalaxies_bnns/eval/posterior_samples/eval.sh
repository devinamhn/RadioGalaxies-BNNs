#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/eval/out-slurm_%j.out


pwd;

nvidia-smi 
echo ">>>start"
source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>evaluating"
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/eval/posterior_samples/combine_analysis.py