#!/bin/bash

#SBATCH --job-name=hamiltorch
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/logs/inits/out-slurm_%A_%a.out
#SBATCH --array=1-5
#SBATCH --cpus-per-task=17
#SBATCH --exclude=compute-0-4

#### #SBATCH --gpus-per-task=1 ###--gres=gpu:1    #SBATCH --exclusive

pwd;

nvidia-smi 
echo ">>>start task array $SLURM_ARRAY_TASK_ID"
echo "Running on:"
hostname
source /share/nas2/dmohan/RadioGalaxies-BNNs/venv/bin/activate
echo ">>>training"
python /share/nas2/dmohan/RadioGalaxies-BNNs/radiogalaxies_bnns/inference/hmc/temp.py $SLURM_ARRAY_TASK_ID

