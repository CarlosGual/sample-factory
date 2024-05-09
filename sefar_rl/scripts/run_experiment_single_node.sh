#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=00:10:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID.out
#$ -N doom_basic_all_envs_sefar

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

python -m sample_factory.launcher.run \
  --run=sefar_rl.experiments.sefar_doom_all_basic_envs \
  --backend=processes \
  --num_gpus=2 \
  --max_parallel=32 \
  --pause_between=0 \
  --experiments_per_gpu=8
