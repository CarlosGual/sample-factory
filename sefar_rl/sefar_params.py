from sample_factory.utils.utils import str2bool


def add_sefar_args(parser):
    p = parser
    p.add_argument(
        "--sparsity",
        default=0.1,
        type=float,
        help="The sparsity for mask to apply to the feature output of the core. Default value is 0.1",
    )
    p.add_argument(
        "--update_mask",
        default=False,
        type=str2bool,
        help="Weather to update the mask or not each forward step. Default value is False")
    p.add_argument(
        "--temp",
        default=1,
        type=int,
        help="Temperature for kl loss in sefar L3 loss function. Default value is 1.",
    )
    p.add_argument(
        "--forward_head",
        default=1,
        type=int,
        help="Which forward head to use. 1 for normal one, 2 for sefar one. Default value is 1.",
    )
    p.add_argument(
        "--weight_kd",
        default=0.1,
        type=float,
        help="Weight given to the knowledge distillation loss (L3). Default value is 0.1.",
    )
    p.add_argument(
        "--with_sefar",
        default=True,
        type=str2bool,
        help="Wether to use SEFAR loss. Default value is True.",
    )
    p.add_argument(
        "--sweep_count",
        default=500,
        type=int,
        help="How many times to run the sweep. Default value is 500.",
    )


def tsubame_override_defaults(parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        train_for_env_steps=50000000,
        algo="APPO",
        env_frameskip=4,
        use_rnn=True,
        num_envs_per_worker=30,
        num_policies=1,
        batch_size=4096,
        wide_aspect_ratio=False,
        num_batches_per_epoch=4,
        with_wandb=True,
        wandb_user="aklab",
        wandb_project="sefar-rl",
    )
