def add_sefar_args(parser):
    p = parser

    p.add_argument(
        "--sparsity",
        default=0.1,
        type=float,
        help="The sparsity for mask to apply to the feature output of the core. Default value is 0.1",
    )
    p.add_argument("--update_mask", default=False, type=bool, help="Wether to update the mask or not each forward step. Default value is False")
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