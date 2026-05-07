#!/bin/bash
#SBATCH --job-name=pist_jepa_train
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk10608@nyu.edu
#SBATCH --output=/scratch/mk10608/logs/train_%j.out
#SBATCH --error=/scratch/mk10608/logs/train_%j.err

mkdir -p /scratch/mk10608/logs
mkdir -p /scratch/mk10608/checkpoints/pist_jepa

singularity exec --nv \
  --overlay /scratch/mk10608/my_env.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "
    source /ext3/env.sh
    conda activate pist_jepa
    cd ~/heirarchical_jepa
    python training/train.py \
      --data_root       /scratch/mk10608/data/active_matter \
      --stats_path      /scratch/mk10608/data/active_matter/stats.json \
      --checkpoint_dir  /scratch/mk10608/checkpoints/pist_jepa \
      --resume          auto \
      --patch_size      16 \
      --embed_dim       384 \
      --encoder_depth   6 \
      --num_heads       6 \
      --pred_dim        192 \
      --num_frames      16 \
      --img_size        224 \
      --use_delta \
      --min_mask_frames 4 \
      --max_mask_frames 8 \
      --use_ema \
      --grad_ckpt \
      --epochs          100 \
      --batch_size      8 \
      --lr              1.5e-4 \
      --min_lr          1e-6 \
      --weight_decay    0.04 \
      --grad_clip       1.0 \
      --warmup_epochs   10 \
      --num_workers     4 \
      --bf16 \
      --seed            42 \
      --wandb_project   pist_jepa_nyu \
      --run_name        pist_jepa_v1 \
      --log_every       50 \
      --val_every       5
  "
