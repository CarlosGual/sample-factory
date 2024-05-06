from __future__ import annotations
import time

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, PolicyID
from sample_factory.utils.utils import log


class SefarLearner(Learner):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids: int):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = masked_select(loss, valids, num_invalids)
        loss = -loss.mean()

        return loss

    def _value_loss(
            self,
            new_values: Tensor,
            old_values: Tensor,
            target: Tensor,
            clip_value: float,
            valids: Tensor,
            num_invalids: int,
    ) -> Tensor:
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = masked_select(value_loss, valids, num_invalids)
        value_loss = value_loss.mean()

        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def _kl_loss(
            self, action_space, action_logits, action_distribution, valids, num_invalids: int
    ) -> Tuple[Tensor, Tensor]:
        old_action_distribution = get_action_distribution(action_space, action_logits)
        kl_old = action_distribution.kl_divergence(old_action_distribution)
        kl_old = masked_select(kl_old, valids, num_invalids)
        kl_loss = kl_old.mean()

        kl_loss *= self.cfg.kl_loss_coeff

        return kl_old, kl_loss

    def _entropy_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        entropy = action_distribution.entropy()
        entropy = masked_select(entropy, valids, num_invalids)
        entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _symmetric_kl_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = masked_select(kl_prior, valids, num_invalids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        # calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            head_outputs = self.actor_critic.forward_head(mb.normalized_obs)

        # initial rnn states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # this is the only way to stop RNNs from backpropagating through invalid timesteps
                # (i.e. experience collected by another policy)
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    head_outputs,
                    done_or_invalid,
                    mb.rnn_states,
                    recurrence,
                )
            else:
                rnn_states = mb.rnn_states[::recurrence]

        # calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = self.actor_critic.forward_core(head_output_seq, rnn_states)
                core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
            else:
                core_outputs, _ = self.actor_critic.forward_core(head_outputs, rnn_states)

        num_trajectories = head_outputs.size(0) // recurrence

        with self.timing.add_time("tail"):
            assert core_outputs.shape[0] == head_outputs.shape[0]

            # calculate policy tail outside of recurrent loop
            result = self.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)
            action_distribution1, action_distribution2 = self.actor_critic.action_distributions()
            log_prob_actions1 = action_distribution1.log_prob(mb.actions)
            ratio1 = torch.exp(log_prob_actions1 - mb.log_prob_actions)  # pi / pi_old

            log_prob_actions2 = action_distribution2.log_prob(mb.actions)
            ratio2 = torch.exp(log_prob_actions2 - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio1 = torch.clamp(ratio1, 0.05, 20.0)
            ratio2 = torch.clamp(ratio2, 0.05, 20.0)

            values1 = result["values1"].squeeze()
            values2 = result["values2"].squeeze()

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio1.cpu()
                values_cpu = values1.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((num_trajectories * recurrence))
                adv = torch.zeros((num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1:: recurrence] - rewards_cpu[recurrence - 1:: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss1 = self._policy_loss(ratio1, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            exploration_loss1 = self.exploration_loss_func(action_distribution1, valids, num_invalids)
            kl_old1, kl_loss1 = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution1, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss1 = self._value_loss(values1, old_values, targets, clip_value, valids, num_invalids)

            policy_loss2 = self._policy_loss(ratio2, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            exploration_loss2 = self.exploration_loss_func(action_distribution2, valids, num_invalids)
            kl_old2, kl_loss2 = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution2, valids, num_invalids
            )
            value_loss2 = self._value_loss(values2, old_values, targets, clip_value, valids, num_invalids)

        return action_distribution1, policy_loss1, exploration_loss1, kl_old1, kl_loss1, value_loss1, \
            action_distribution2, policy_loss2, exploration_loss2, kl_old2, kl_loss2, value_loss2, locals()

    def _train(
            self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = torch.empty([self.cfg.num_batches_per_epoch], device=self.device)

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                        self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution1,
                        policy_loss1,
                        exploration_loss1,
                        kl_old1,
                        kl_loss1,
                        value_loss1,
                        action_distribution2,
                        policy_loss2,
                        exploration_loss2,
                        kl_old2,
                        kl_loss2,
                        value_loss2,
                        loss_locals,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss1: Tensor = policy_loss1 + exploration_loss1 + kl_loss1
                    critic_loss1 = value_loss1
                    ppo_loss1: Tensor = actor_loss1 + critic_loss1

                    actor_loss2: Tensor = policy_loss2 + exploration_loss2 + kl_loss2
                    critic_loss2 = value_loss2
                    ppo_loss2: Tensor = actor_loss2 + critic_loss2

                    epoch_actor_losses[batch_num] = (actor_loss1 + actor_loss2)/2

                    divergence_loss = torch.nn.KLDivLoss(reduction="batchmean")
                    # Detach action_distribution2
                    # action_distribution2 = action_distribution2.
                    kd_loss = divergence_loss(action_distribution1.log_probs/self.cfg.temp, action_distribution2.log_probs.detach()/self.cfg.temp)
                    loss = ppo_loss1 + ppo_loss2 + self.cfg.weight_kd * kd_loss

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss1 + policy_loss2),
                            to_scalar(value_loss1 + value_loss2),
                            to_scalar(exploration_loss1 + exploration_loss2),
                            to_scalar(kl_loss1 + kl_loss2),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None, it is already calculated above
                    if kl_old1 is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old1 = action_distribution1.kl_divergence(old_action_distribution)
                        kl_old1 = masked_select(kl_old1, mb.valids, num_invalids)
                        kl_old2 = action_distribution2.kl_divergence(old_action_distribution)
                        kl_old2 = masked_select(kl_old2, mb.valids, num_invalids)

                    kl_old_mean = (kl_old1.mean().item() + kl_old2.mean().item())/2
                    recent_kls.append(kl_old_mean)
                    if kl_old1.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old1.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**loss_locals, **locals()}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = epoch_actor_losses.mean().item()
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.lr = self.curr_lr
        stats.actual_lr = train_loop_vars.actual_lr  # potentially scaled because of masked data

        stats.update(self.actor_critic.summaries())

        stats.valids_fraction = var.mb.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self.policy_id).float().mean()

        grad_norm = (
                sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if
                    p.grad is not None) ** 0.5
        )
        stats.grad_norm = grad_norm
        stats.loss = var.loss
        stats.value1 = var.result["values1"].mean()
        stats.entropy1 = var.action_distribution1.entropy().mean()
        stats.policy_loss1 = var.policy_loss1
        stats.kl_loss1 = var.kl_loss1
        stats.value_loss1 = var.value_loss1
        stats.exploration_loss1 = var.exploration_loss1

        stats.value2 = var.result["values2"].mean()
        stats.entropy2 = var.action_distribution2.entropy().mean()
        stats.policy_loss2 = var.policy_loss2
        stats.kl_loss2 = var.kl_loss2
        stats.value_loss2 = var.value_loss2
        stats.exploration_loss2 = var.exploration_loss2

        stats.adv_min = var.adv.min()
        stats.adv_max = var.adv.max()
        stats.adv_std = var.adv_std
        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        if hasattr(var.action_distribution1, "summaries"):
            stats.update(var.action_distribution1.summaries())
        if hasattr(var.action_distribution2, "summaries"):
            stats.update(var.action_distribution2.summaries())

        if var.epoch == self.cfg.num_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
            valid_ratios = masked_select((var.ratio1+var.ratio2)/2, var.mb.valids, var.num_invalids)
            ratio_mean = torch.abs(1.0 - valid_ratios).mean().detach()
            ratio_min = valid_ratios.min().detach()
            ratio_max = valid_ratios.max().detach()
            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

            value_delta = torch.abs((var.values1+var.values2)/2 - var.old_values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            stats.kl_divergence = var.kl_old_mean
            # stats.kl_divergence_max = var.kl_old.max()
            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max
            # noinspection PyUnresolvedReferences
            stats.fraction_clipped = (
                    (valid_ratios < var.clip_ratio_low).float() + (valid_ratios > var.clip_ratio_high).float()
            ).mean()
            stats.ratio_mean = ratio_mean
            stats.ratio_min = ratio_min
            stats.ratio_max = ratio_max
            stats.num_sgd_steps = var.num_sgd_steps

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            adam_max_second_moment = max(tensor_state["exp_avg_sq"].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self.policy_id]
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.summaries_use_frameskip:
                self.env_steps += experience_size * self.env_info.frameskip
            else:
                self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats
