#!/bin/bash

#SBATCH --job-name=dataloading
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/mcmc/hamilt/logs/dataloading/out-slurm_%j.out
#SBATCH --exclude=compute-0-116,compute-0-117,compute-0-118,compute-0-119,compute-0-4


pwd;

nvidia-smi 
echo ">>>start"
source /share/nas2/dmohan/mcmc/hamilt/venv/bin/activate
echo ">>>training"
python /share/nas2/dmohan/mcmc/hamilt/combine.py

