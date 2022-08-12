#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH -a 1-4%1
#SBATCH --j lantemporalgan
#SBATCH -o /HPS/HumanBodyRetargeting7/work/slurm_logs/pix2pixhd-%j.out


python train.py \
--config /HPS/HumanBodyRetargeting2/work/Code/Neural-Actor/scripts/texture_generation/DeepcapLan_vid2vid_follow_dancing_bz4_temporalgan.yaml \
--logdir /HPS/HumanBodyRetargeting2/work/Release_data_example/vidvid_checkpoint \
--single_gpu
