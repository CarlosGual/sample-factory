#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=08:00:00
#$ -o slurm_logs/$JOB_NAME_$JOB_ID.out
#$ -N sweep_test_doom_defend_the_center

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

exp_name="sweep_test"
env_name="doom_defend_the_center"
EXPERIMENT="${exp_name}_${env_name}"
num_cpus=$(nproc)

python -m sefar_rl.train_sefar_with_sweeps \
 --env $env_name \
 --train_dir train_dir \
 --restart_behavior overwrite \
 --algo APPO \
 --serial_mode False \
 --env_frameskip 4 \
 --use_rnn True \
 --worker_num_splits 2 \
 --num_envs_per_worker 30 \
 --num_workers "$num_cpus" \
 --num_policies 1 \
 --batch_size 4096 \
 --num_batches_per_epoch 4 \
 --experiment $EXPERIMENT \
 --res_w 128 \
 --res_h 72 \
 --wide_aspect_ratio False \
 --policy_workers_per_policy 2 \
 --with_sefar True \
 --train_for_env_steps 200000000 \
 --with_wandb True \
 --wandb_user aklab \
 --wandb_project sefar-rl \
 --wandb_tags doom appo with_sefar \
 --sweep_count 500
