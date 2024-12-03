#!/bin/bash
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*56)
#SBATCH --ntasks=1
#! How much wallclock time will be required?
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --reservation=gpu_hackathon
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.out
#SBATCH --account=MANAPY-1WABCJWE938-DEFAULT-GPU

./cholesky
