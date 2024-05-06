#!/bin/bash
#AKBATCH -r drakee_2
#SBATCH -N 1
#SBATCH --requeue
#SBATCH -J doom_test
#SBATCH --output=slurm_logs/%x-%j.out

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

exp_name="doom_test"
env_name="doom_benchmark"
EXPERIMENT="${exp_name}_${env_name}"

python -m sefar_rl.train_sefar \
 --env $env_name \
 --train_dir pruebas \
 --algo APPO \
 --serial_mode False \
 --env_frameskip 4 \
 --use_rnn True \
 --num_workers 20 \
 --num_envs_per_worker 32 \
 --num_policies 1 \
 --batch_size 4096 \
 --experiment $EXPERIMENT \
 --res_w 128 \
 --res_h 72 \
 --wide_aspect_ratio False \
 --policy_workers_per_policy 2 \
 --worker_num_splits 2 \
 --with_wandb True \
 --wandb_user aklab \
 --wandb_project sefar-rl \
 --wandb_tags test doom appo


