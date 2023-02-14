#!/bin/bash
#SBATCH --account m4277
#SBATCH --constraint gpu
#SBATCH --qos regular
#SBATCH --time 12:00:00
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --gpus-per-task 1
#SBATCH --output=slurm/logs/benchmarks-%x.%j.out

export SLURM_CPU_BIND="cores"

module load python
source activate sqt

srun python benchmarks.py

# perform any cleanup or short post-processing here
