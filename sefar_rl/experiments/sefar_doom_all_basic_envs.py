from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "doom_my_way_home",
                "doom_deadly_corridor",
                "doom_defend_the_center",
                "doom_defend_the_line",
                "doom_health_gathering",
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
        "basic_envs_sefar",
        "python -m sefar_rl.train_sefar --train_for_env_steps=50000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_envs_per_worker=30 --num_policies=1 --batch_size=4096 --wide_aspect_ratio=False --num_batches_per_epoch=4 --with_wandb=True --wandb_user=aklab --wandb_project=sefar-rl --wandb_tags doom",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("sefar_doom_basic_envs", experiments=_experiments)
