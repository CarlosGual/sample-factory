#!/usr/bin/env python3
"""This is an example to train PPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import torch
import wandb

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import RaySampler, VecWorker
from garage.torch import set_gpu_mode
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@click.option('--with_wandb', default=True)
@wrap_experiment(snapshot_mode='all')
def mtppo_metaworld_mt1_push(ctxt, seed, epochs, batch_size, with_wandb):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    if with_wandb:
        wandb.init(project="sefar-mw",
                   name=f'mtppo_metaworld_mt1_push_{seed}_{epochs}_{batch_size}',
                   sync_tensorboard=True,
                   config=None,
                   entity='aklab',
                   )
    set_seed(seed)
    n_tasks = 50
    mt1 = metaworld.MT1('push-v1')
    train_task_sampler = MetaWorldTaskSampler(mt1, 'train',
                                              lambda env, _: normalize(env))
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         worker_class=VecWorker,
                         worker_args=dict(n_envs=12)
                         )

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)

    # enable GPU
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    algo.to()

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


mtppo_metaworld_mt1_push()
