#!/bin/bash

#SBATCH --job-name=energy
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/ood/out-slurm_%j.out


pwd;

nvidia-smi 
echo ">>>start"
source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>evaluating" 
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/eval/ood/hmc_ood.py #laplace_ood.py #ensembles_ood.py #dropout_ood.py