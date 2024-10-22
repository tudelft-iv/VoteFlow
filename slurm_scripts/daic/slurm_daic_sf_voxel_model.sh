#!/bin/bash

#SBATCH --job-name="sf_voxel_model_with_supervised_loss"
#SBATCH --partition=cor 
#SBATCH --time=48:00:00

#SBATCH --account=me-cor 
#SBATCH --qos=reservation 
#SBATCH --reservation=cvpr-cor

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1

#SBATCH --mail-type=END
#SBATCH --output=slurm_logs/slurm_sf_voxel_model_with_supervised_loss_%j.out
#SBATCH --error=slurm_logs/slurm_sf_voxel_model_with_supervised_loss_%j.err

module load cuda/11.8
module load devtoolset/11

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

conda activate sf_tv

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

nvidia-smi

srun python train.py model=sf_voxel_model lr=2e-5 epochs=20 batch_size=8 loss_fn=ff3dLoss wandb_mode=online exp_note="with_ff3dLoss"

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
