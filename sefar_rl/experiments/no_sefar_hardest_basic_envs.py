from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "doom_deadly_corridor",
                "doom_defend_the_line",
                "doom_health_gathering_supreme",
            ],
        ),
        # ("sparsity", [0.1, 0.5, 0.9]),
        # ("update_mask", [True, False]),
        # ("temp", [1, 5, 10]),
        # ("weight_kd", [0.1, 1, 10]),
        # ("with_sefar", [True, False]),
    ]
)

_experiments = [
    Experiment(
        "no_sefar_hardest_basic_envs",
        "python -m sefar_rl.train_resnet --restart_behavior overwrite --train_dir train_dir "
        "--train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers 48 "
        "--num_envs_per_worker=30 --worker_num_splits 2 --num_policies=1 --batch_size=4096 --wide_aspect_ratio=False "
        "--res_w 128 --res_h 72 --num_batches_per_epoch=4 --with_sefar False --with_wandb=False --wandb_user=aklab "
        "--wandb_project=sefar-rl --wandb_tags doom appo no_sefar",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("no_sefar_hardest_basic_envs", experiments=_experiments)
