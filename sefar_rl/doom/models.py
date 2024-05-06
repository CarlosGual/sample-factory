from typing import List

import torch
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder, make_img_encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.model_utils import nonlinearity, create_mlp
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


class VizdoomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        # reuse the default image encoder
        self.basic_encoder = make_img_encoder(cfg, obs_space["obs"])
        self.encoder_out_size = self.basic_encoder.get_out_size()

        self.measurements_head = None
        if "measurements" in obs_space.keys():
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_space["measurements"].shape[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_space["measurements"].shape)
            self.encoder_out_size += measurements_out_size

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict["measurements"].float())
            x = torch.cat((x, measurements), dim=1)

        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_vizdoom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return VizdoomEncoder(cfg, obs_space)


class SefarDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        self.core_input_size = decoder_input_size
        decoder_layers: List[int] = cfg.decoder_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(decoder_layers, decoder_input_size, activation)
        if len(decoder_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        self.decoder_out_size = calc_num_elements(self.mlp, (decoder_input_size,))

    def forward(self, core_output):
        return self.mlp(core_output)

    def get_out_size(self):
        return self.decoder_out_size


def sefar_make_decoder_func(cfg: Config, core_input_size: int) -> Decoder:
    return SefarDecoder(cfg, core_input_size)
