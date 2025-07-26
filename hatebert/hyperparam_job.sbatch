#!/bin/bash
#SBATCH --job-name=hparam-v100
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o ./slurm_outs/slurm_%j.out 

module load anaconda3/2022.05.0.1
module load cuda

conda activate /home/hice1/aamir37/scratch/cs7643/hatebert

python hyperparam_search.py