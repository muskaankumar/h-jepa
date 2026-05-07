# Environment Setup (NYU HPC)

## Prerequisites
- Access to `ood-burst-001.hpc.nyu.edu` (NYU VPN required off-campus)
- Slurm account: `csci_ga_2572-2026sp`

## Step 1 — Create Singularity overlay

```bash
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3 /scratch/$USER/my_env.ext3

singularity exec \
  --overlay /scratch/$USER/my_env.ext3:rw \
  /share/apps/images/cuda12.1-cudnn8.7-devel-ubuntu22.04.sif \
  /bin/bash
```

Inside the container:
```bash
# Create env bootstrap script
mkdir -p /ext3
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
unset -f which
source /root/miniconda3/etc/profile.d/conda.sh
EOF

# Install miniconda if not present
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /root/miniconda3

source /root/miniconda3/etc/profile.d/conda.sh
conda create -n pist_jepa python=3.10 -y
conda activate pist_jepa

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install einops timm h5py wandb huggingface_hub datasets numpy scipy
```

## Step 2 — Download dataset

```bash
# Submit as a CPU batch job or run in tmux
singularity exec --overlay /scratch/$USER/my_env.ext3:ro \
  /share/apps/images/cuda12.1-cudnn8.7-devel-ubuntu22.04.sif \
  bash -c "
    source /ext3/env.sh && conda activate pist_jepa
    huggingface-cli download polymathic-ai/active_matter \
      --repo-type dataset \
      --local-dir /scratch/$USER/data/active_matter
  "
```

## Step 3 — Compute normalization stats (once)

```bash
singularity exec --nv --overlay /scratch/$USER/my_env.ext3:ro \
  /share/apps/images/cuda12.1-cudnn8.7-devel-ubuntu22.04.sif \
  bash -c "
    source /ext3/env.sh && conda activate pist_jepa
    python ~/pist_jepa/data/compute_stats.py \
      --data_root /scratch/$USER/data/active_matter \
      --output    /scratch/$USER/data/active_matter/stats.json
  "
```

## Step 4 — Train

```bash
sbatch ~/pist_jepa/scripts/train_a100.sh
```

## Step 5 — Evaluate

```bash
sbatch ~/pist_jepa/scripts/eval.sh
```

## Monitoring

```bash
squeue -u $USER                 # job status
tail -f /scratch/$USER/logs/train_<JOB_ID>.out  # live log
```

## Notes
- All checkpoints saved to `/scratch/$USER/checkpoints/pist_jepa/`
- `ckpt_latest.pt` — most recent epoch (for preemption recovery)
- `ckpt_best.pt`   — best validation loss
- W&B dashboard at https://wandb.ai (set WANDB_API_KEY in your environment)
