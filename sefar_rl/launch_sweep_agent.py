import functools
import sys

import wandb

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sefar_rl.model.resnet_model import make_vizdoom_resnet_encoder
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
    global_model_factory().register_encoder_factory(make_vizdoom_resnet_encoder)
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
    parser.add_argument(
        "--sweep_id",
        type=str,
        help="sweep_id to add more wandb agents to the sweep",
    )
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_vizdoom_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    cfg = parse_vizdoom_cfg()
    sys.exit(wandb.agent(sweep_id=cfg.sweep_id, function=main, count=cfg.sweep_count))
