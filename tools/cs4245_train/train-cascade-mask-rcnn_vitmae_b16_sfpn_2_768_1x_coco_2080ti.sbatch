#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32768
#SBATCH --gres=gpu:turing:1
#SBATCH --mail-type=END

cd ../../

srun python3 tools/train.py configs/cascade_rcnn/cascade-mask-rcnn_vitmae_b16_sfpn_2_768_1x_coco.py