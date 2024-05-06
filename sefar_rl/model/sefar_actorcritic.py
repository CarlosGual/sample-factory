from __future__ import annotations

from typing import Dict

import torch
from sample_factory.model.actor_critic import ActorCritic
from torch import Tensor, nn

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.algo.utils.action_distributions import sample_actions_log_probs


class SefarActorCritic(ActorCritic):
    def __init__(
            self,
            model_factory,
            cfg: Config,
            obs_space: ObsSpace,
            action_space: ActionSpace,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())

        # TODO: revise core feature output size
        mask = torch.rand(self.core.get_out_size())
        mask[mask <= self.cfg.sparsity] = 0.
        mask[mask != 0] = 1.
        self.mask = mask

        self.decoder1 = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder1_out_size: int = self.decoder1.get_out_size()

        self.decoder2 = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder2_out_size: int = self.decoder2.get_out_size()

        self.critic_linear1 = nn.Linear(decoder1_out_size, 1)
        self.action_parameterization1 = self.get_action_parameterization(decoder1_out_size)

        self.critic_linear2 = nn.Linear(decoder2_out_size, 1)
        self.action_parameterization2 = self.get_action_parameterization(decoder2_out_size)

        self.last_action_distribution1 = None
        self.last_action_distribution2 = None

        self.apply(self.initialize_weights)

    def model_to_device(self, device):
        for module in self.children():
            # allow parts of encoders/decoders to be on different devices
            # (i.e. text-encoding LSTM for DMLab is faster on CPU)
            if hasattr(module, "model_to_device"):
                module.model_to_device(device)
            else:
                module.to(device)
        # Add mask to device
        self.mask = self.mask.to(device)

    def action_distributions(self):
        return self.last_action_distribution1, self.last_action_distribution2

    def _update_mask(self, feature_shape):
        mask = torch.rand(feature_shape, device=self.device)
        mask[mask <= self.cfg.sparsity] = 0.
        mask[mask != 0] = 1.
        return mask

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        if self.cfg.update_mask:
            self.mask = self._update_mask(core_output.shape)

        core_output_sparse = core_output * self.mask

        decoder_output1 = self.decoder1(core_output)
        decoder_output2 = self.decoder2(core_output_sparse)

        values1 = self.critic_linear1(decoder_output1).squeeze()
        values2 = self.critic_linear2(decoder_output2).squeeze()

        if self.cfg.forward_head == 1:
            values = values1
        elif self.cfg.forward_head == 2:
            values = values2

        result = TensorDict(values=values, values1=values1, values2=values2)
        if values_only:
            return result

        action_distribution_params1, self.last_action_distribution1 = self.action_parameterization1(decoder_output1)
        action_distribution_params2, self.last_action_distribution2 = self.action_parameterization2(decoder_output2)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        if self.cfg.forward_head == 1:
            result["action_logits"] = action_distribution_params1
        elif self.cfg.forward_head == 2:
            result["action_logits"] = action_distribution_params2

        self._maybe_sample_actions(sample_actions, result) ## SEEMS TO NOT BE USED
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            if self.cfg.forward_head == 1:
                actions, result["log_prob_actions"] = sample_actions_log_probs(self.last_action_distribution1)
            elif self.cfg.forward_head == 2:
                actions, result["log_prob_actions"] = sample_actions_log_probs(self.last_action_distribution2)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)

def make_sefar_actorcritic(model_factory, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> SefarActorCritic:
    return SefarActorCritic(model_factory, cfg, obs_space, action_space)
