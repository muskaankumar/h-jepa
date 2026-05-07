#!/bin/bash
# ============================================================
# PIST-JEPA Evaluation Job — Linear Probe + kNN
# ============================================================
#SBATCH --job-name=pist_jepa_eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/eval_%j.out
#SBATCH --error=/scratch/%u/logs/eval_%j.err

set -euo pipefail
echo "Eval job $SLURM_JOB_ID starting at $(date)"

SIF=/share/apps/images/cuda12.1-cudnn8.7-devel-ubuntu22.04.sif
OVL=/scratch/$USER/my_env.ext3
DATA=/scratch/$USER/data/active_matter
CKPT=/scratch/$USER/checkpoints/pist_jepa/ckpt_best.pt
CODE=$HOME/pist_jepa
OUT=/scratch/$USER/eval_results/pist_jepa

singularity exec --nv \
  --overlay ${OVL}:ro \
  ${SIF} \
  bash -c "
    source /ext3/env.sh
    conda activate pist_jepa

    python ${CODE}/eval/eval_probe.py \
      --checkpoint    ${CKPT} \
      --data_root     ${DATA} \
      --stats_path    ${DATA}/stats.json \
      --output_dir    ${OUT} \
      --patch_size    16 \
      --embed_dim     384 \
      --encoder_depth 6 \
      --num_heads     6 \
      --num_frames    16 \
      --img_size      224 \
      --use_delta \
      --probe_epochs  100 \
      --probe_lr      1e-3
  "

echo "Eval done at $(date)"
