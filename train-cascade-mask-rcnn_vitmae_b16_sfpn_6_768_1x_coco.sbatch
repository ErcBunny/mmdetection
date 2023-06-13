#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --gre=gpu:a40:1
#SBATCH --mail-type=END

srun python3 tools/train.py configs/cascade_rcnn/cascade-mask-rcnn_vitmae_b16_sfpn_6_768_1x_coco.py