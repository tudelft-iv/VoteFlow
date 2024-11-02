#!/bin/bash

#SBATCH --job-name="sf_voxel_model_with_seloss_new_dst_radius_m_32_n_64_src_knn"
#SBATCH --partition=cor 
#SBATCH --time=72:00:00

#SBATCH --account=me-cor 
#SBATCH --qos=reservation 
#SBATCH --reservation=cvpr-cor

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gpus-per-task=8

#SBATCH --mail-type=END
#SBATCH --output=slurm_logs/slurm_sf_voxel_model_with_seloss_new_dst_radius_m_32_n_64_src_knn_%j.out
#SBATCH --error=slurm_logs/slurm_sf_voxel_model_with_seloss_new_dst_radius_m_32_n_64_src_knn_%j.err

module load cuda/11.8
module load devtoolset/11
module load miniconda/3.9

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export WANDB_DIR=/home/nfs/shimingwang/workspace/sceneflow_tv/.wandb
export WANDB_CACHE_DIR=/home/nfs/shimingwang/workspace/sceneflow_tv/.wandb/cache

conda activate sf_tv

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

nvidia-smi

srun python train.py model=sf_voxel_model lr=2e-4 epochs=20 batch_size=4 model.target.use_bn_in_vol=True model.target.m=32 model.target.n=64 loss_fn=seflowLoss exp_note="with_seflowLoss_new_dst_radius_m_32_n_64" wandb_mode="online" gpus="auto" "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" "model.val_monitor=val/Dynamic/Mean"

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
