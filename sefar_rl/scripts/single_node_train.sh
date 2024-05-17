#!/bin/bash
#AKBATCH -r drakee_2
#SBATCH -N 1
#SBATCH --requeue
#SBATCH -J doom_test
#SBATCH --output=slurm_logs/%x-%j.out

# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sefar-rl

exp_name="testeos"
env_name="doom_health_gathering_supreme"
EXPERIMENT="${exp_name}_${env_name}"
num_cpus=2 #$(nproc)

python -m sefar_rl.train_sefar_resnet \
 --update_mask True \
 --with_sefar True \
 --env $env_name \
 --train_dir pruebas \
 --algo APPO \
 --serial_mode True \
 --restart_behavior overwrite \
 --train_for_env_steps 500000000 \
 --env_frameskip 4 \
 --use_rnn True \
 --worker_num_splits 2 \
 --num_envs_per_worker 2 \
 --num_workers "$num_cpus" \
 --update_mask False \
 --num_policies 1 \
 --batch_size 2048 \
 --num_batches_per_epoch 4 \
 --experiment $EXPERIMENT \
 --res_w 128 \
 --res_h 72 \
 --wide_aspect_ratio False \
 --policy_workers_per_policy 2 \
 --with_wandb True \
 --wandb_user aklab \
 --wandb_project sefar-rl
# --wandb_tags test doom appo
