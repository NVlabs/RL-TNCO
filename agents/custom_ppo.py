from collections import defaultdict
from typing import Union, Type, Optional, Dict, Any, List

import numpy as np
import torch as th
from gym.vector.utils import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch.nn import functional

from agents.custom_buffer import OptimisticRolloutBuffer, DictRolloutBufferModified
from gnn.gnn_model import GNN_model


def get_policy_loss(advantages, clip_range, log_prob, rollout_data):
    # ratio between old and new policy, should be one at the first iteration
    ratio = th.exp(log_prob - rollout_data.old_log_prob)
    # clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
    return policy_loss


def get_entropy_loss(entropy, log_prob):
    if entropy is None:
        # Approximate entropy when no analytical form
        entropy_loss = -th.mean(-log_prob)
    else:
        entropy_loss = -th.mean(entropy)
    return entropy_loss


class CuQuantumPPO(PPO):

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy], Type[GNN_model]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            # custom params
            policy_base: Type[BasePolicy] = ActorCriticPolicy
            ):
        super().__init__(
                policy=policy,
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                normalize_advantage=normalize_advantage,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                target_kl=target_kl,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                seed=seed,
                device=device,
                _init_setup_model=_init_setup_model,
                )

        if _init_setup_model:
            self.rollout_buffer = DictRolloutBufferModified(
                    self.n_steps,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    n_envs=self.n_envs,
                    )

        self.config = self.policy_kwargs['config'] if self.policy_kwargs is not None else None
        self.ep_cost_buffer = None
        self.episode_total_return, self.episode_median_return, self.mean_episode_evaluated_return = None, None, None
        self.best_results = None
        self.use_optimistic_buffer = self.config['learning']['use_optimistic_buffer']
        self.info_buffer = self.reset_info_buffer()
        if self.config['learning']['use_optimistic_buffer']:
            self.optimistic_epochs = self.config['learning']['optimistic_epochs']
            self.optimistic_buffer = OptimisticRolloutBuffer(
                    self.config['learning']['optimistic_buffer_size'],
                    self.observation_space,
                    self.action_space,
                    self.device,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    )

    def train(self) -> None:
        """
		Update policy using the currently gathered rollout buffer.
		"""
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        extra_metrics = dict()
        advantage_log = []
        abs_advantage_log = []
        log_struct = defaultdict(list)
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # this is off by default
                if self.config['learning']['normalize_advantage']:
                    mean_advantage = advantages.mean()
                    advantage_std = (advantages.std() + 1e-8)
                    advantages = (advantages - mean_advantage) / advantage_std

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                value_loss = self.get_value_loss(rollout_data, values)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                entropy_loss = get_entropy_loss(entropy, log_prob)

                entropy_losses.append(entropy_loss.item())
                # update_metrics(log_struct, new_info=advantages.mean(), key='advantages')
                # update_metrics(log_struct, new_info=advantages.abs().mean(), key='abs_advantages')
                abs_advantage_log.append(advantages.abs().mean())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.policy.optimizer.step()

            if not continue_training:
                break
        if self.use_optimistic_buffer:
            self.update_optimistic_buffer()
        if self.use_optimistic_buffer and self.optimistic_buffer.pos > 0 and \
                (self.config['learning']['update_policy_using_optimistic_buffer'] or
                 self.config['learning']['update_v_using_optimistic_buffer']):
            for epoch in range(self.optimistic_epochs):
                # Do a complete pass on the rollout buffer
                for rollout_data in self.optimistic_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # if that line is commented (as in SAC)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.returns - values.detach()
                    if self.config['learning']['normalize_advantage']:
                        convex_factor = self.config['learning']['convex_factor_for_adv_norm']
                        adv_factor = convex_factor * mean_advantage + (1 - convex_factor) * advantages.min()
                        advantages = (advantages - adv_factor) / advantage_std

                    if self.config['learning']['update_policy_using_optimistic_buffer']:
                        policy_loss = get_policy_loss(advantages, clip_range, log_prob, rollout_data)
                    else:
                        policy_loss = 0

                    if self.config['learning']['update_v_using_optimistic_buffer']:
                        value_loss = self.get_value_loss(rollout_data, values)
                    else:
                        value_loss = 0

                    loss = policy_loss + self.vf_coef * value_loss

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                    self.policy.optimizer.step()

                if not continue_training:
                    break

        self._n_updates += self.n_epochs

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/abs_advantage", np.mean(th.tensor(abs_advantage_log, device='cpu').numpy()))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def get_value_loss(self, rollout_data, values):
        values_pred = values
        # Value loss using the TD(gae_lambda) target
        returns = th.clamp(rollout_data.returns, -self.config['train_params']['clip_return'], 0.0)
        if self.config['learning']['v_norm'] == 2:
            value_loss = functional.mse_loss(returns, values_pred)
        else:
            value_loss = functional.l1_loss(returns, values_pred)
        return value_loss

    def update_optimistic_buffer(self):
        candidate_limit = np.zeros(self.rollout_buffer.episode_starts.shape[1], dtype=int)
        indices = np.ones(self.rollout_buffer.episode_starts.shape, dtype=bool)
        for i in range(self.rollout_buffer.episode_starts.shape[1]):
            episode_start = np.where(self.rollout_buffer.episode_starts[:, i])[0]
            candidate_limit[i] = episode_start[-1] if len(episode_start) > 0 else 0
            indices[candidate_limit[i]:, i] = False
        with th.no_grad():
            on_policy_buffer_scores = self.get_buffer_scores(self.rollout_buffer, indices=indices)
            optimistic_buffer_scores = self.get_buffer_scores(self.optimistic_buffer)
            score = np.concatenate([on_policy_buffer_scores, optimistic_buffer_scores])
            buf_size = self.optimistic_buffer.buffer_size

            if np.sum(on_policy_buffer_scores > 0) + np.sum(optimistic_buffer_scores > 0) > buf_size:
                samples = np.random.choice(np.arange(len(score)), replace=False, p=score / score.sum(), size=buf_size)
                indices_from_on_policy_buffer = samples[samples < len(on_policy_buffer_scores)]
                indices_from_optimistic_buffer = samples[samples >= len(on_policy_buffer_scores)] - len(
                        on_policy_buffer_scores)
            else:
                indices_from_on_policy_buffer = np.zeros_like(on_policy_buffer_scores, dtype=bool)
                indices_from_optimistic_buffer = np.zeros_like(optimistic_buffer_scores, dtype=bool)
                indices_from_on_policy_buffer[on_policy_buffer_scores > 0] = True
                indices_from_optimistic_buffer[optimistic_buffer_scores > 0] = True
            self.optimistic_buffer.add(indices_from_optimistic_buffer, self.rollout_buffer,
                                       indices_from_on_policy_buffer)

    def get_buffer_scores(self, buffer, indices=None):
        data = buffer.get(randomize=False).__next__()
        if data is not None:
            buffer_values, _, _ = self.policy.evaluate_actions(data.observations, actions=None, value_only=True)
            buffer_advs = (data.returns.cpu() - buffer_values.cpu()).numpy()
            score = np.zeros_like(buffer_advs)
            if indices is not None:
                candidate_updates = np.logical_and(buffer_advs > 0, indices.swapaxes(0, 1).flatten())
            else:
                candidate_updates = buffer_advs > 0
            score[candidate_updates] = buffer_advs[candidate_updates]
        else:
            score = np.array([])
        return score

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
		Retrieve reward, episode length, episode success and update the buffer
		if using Monitor wrapper or a GoalEnv.

		:param infos: List of additional information about the transition.
		:param dones: Termination signals
		"""
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            cost_at_0 = info.get("cost_at_0")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if cost_at_0 is not None:
                self.ep_cost_buffer.append(cost_at_0)
            self.info_buffer.append(info)

    def reset_info_buffer(self):
        self.info_buffer = []
        self.ep_cost_buffer = []
        return self.info_buffer
