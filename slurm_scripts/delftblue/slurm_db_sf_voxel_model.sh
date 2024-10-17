#!/bin/bash

#SBATCH --job-name="sf_voxel_model_lower_lr_1e-5"
#SBATCH --partition=gpu-a100
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --account=research-ME-cor
#SBATCH --mail-type=END
#SBATCH --output=slurm_logs/slurm_${SLURM_JOB_NAME}_%j.out
#SBATCH --error=slurm_logs/slurm_${SLURM_JOB_NAME}_%j.err

module load cuda/11.7

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sf_tv

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

nvidia-smi

srun python train.py lr=2e-5 epochs=20 batch_size=8 loss_fn=warpedLoss gpus=[0,1,2,3] wandb_mode=online exp_note="lower_lr_1e-5"

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
