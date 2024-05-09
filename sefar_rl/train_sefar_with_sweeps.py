import functools
import sys

import wandb

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sefar_rl.model.sefar_actorcritic import make_sefar_actorcritic
from sefar_rl.sefar_params import add_sefar_args, tsubame_override_defaults


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)
    global_model_factory().register_actor_critic_factory(make_sefar_actorcritic)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # parameters specific to SEFAR
    tsubame_override_defaults(parser)
    add_sefar_args(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def create_sweep_conf(config):
    sweep_conf = {
        "name": config.experiment,
        "entity": "aklab",
        "project": "sefar-rl",
        "method": "bayes",
        "metric": {"name": "reward/reward", "goal": "maximize"},
        "parameters": {
            "sparsity": {"min": 0.1, "max": 0.9},
            "update_mask": {"values": [True, False]},
            "temp": {"min": 1, "max": 10},
            "weight_kd": {"min": 0.1, "max": 10.0},
            "forward_head": {"values": [1, 2]},
        },
    }
    return sweep_conf


def main():  # pragma: no cover
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_vizdoom_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    cfg = parse_vizdoom_cfg()
    sweep_id = wandb.sweep(create_sweep_conf(cfg))
    sys.exit(wandb.agent(sweep_id=sweep_id, function=main, count=cfg.sweep_count))
