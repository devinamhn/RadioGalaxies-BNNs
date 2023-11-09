#!/bin/bash

#SBATCH --job-name=dropout
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/mcmc/hamilt/dropout/logs/out-slurm_%j.out


pwd;

nvidia-smi 
echo ">>>start"
source /share/nas2/dmohan/mcmc/hamilt/venv/bin/activate
echo ">>>training"
python /share/nas2/dmohan/mcmc/hamilt/dropout.py
