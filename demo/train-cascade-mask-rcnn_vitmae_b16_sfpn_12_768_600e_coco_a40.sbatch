#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32768
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-type=END

srun python3 vitdet_small_train_demo_lowlr_b16.py