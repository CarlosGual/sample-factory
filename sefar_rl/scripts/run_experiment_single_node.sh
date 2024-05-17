#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID.out
#$ -N yes_sefar_battle2

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

python -m sample_factory.launcher.run \
  --run=sefar_rl.experiments.yes_sefar_battle \
  --backend=processes \
  --num_gpus=4 \
  --max_parallel=4 \
  --pause_between=1 \
  --experiments_per_gpu=1
