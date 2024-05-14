#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID.out
#$ -N no_sefar_hardest_basic_envs

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

python -m sample_factory.launcher.run \
  --run=sefar_rl.experiments.no_sefar_hardest_basic_envs \
  --backend=processes \
  --num_gpus=4 \
  --max_parallel=8 \
  --pause_between=0 \
  --experiments_per_gpu=2
