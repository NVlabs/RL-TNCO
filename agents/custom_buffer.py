from typing import Union, Generator, Optional

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples


class DictRolloutBufferModified(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            ):

        super(DictRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                                                gae_lambda=gae_lambda, )

    def get(self, batch_size: Optional[int] = None, randomize=True) -> Generator[DictRolloutBufferSamples, None, None]:
        if randomize:
            indices = np.random.permutation(self.buffer_size * self.n_envs)
            assert self.pos == self.buffer_size
        else:
            indices = np.arange(self.buffer_size * self.n_envs)
        self.set_indices_and_generator()

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def set_indices_and_generator(self):
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True


def num_of_elements(indices):
    if indices.dtype == bool:
        return np.sum(indices)
    else:
        return len(indices)


class OptimisticRolloutBuffer(DictRolloutBufferModified):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            ):
        super(OptimisticRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device,
                                                      gae_lambda=gae_lambda,
                                                      gamma=gamma,
                                                      n_envs=1)

    def add(self, local_indices, external_buffer, external_indices) -> None:
        for key in self.observations.keys():
            self.observations[key] = np.concatenate([self.observations[key][local_indices],
                                                     external_buffer.observations[key][external_indices].copy()])

        self.actions = np.concatenate([self.actions[local_indices], external_buffer.actions[external_indices].copy()])
        self.rewards = np.concatenate([self.rewards[local_indices],
                                       self.swap_and_flatten(external_buffer.rewards)[external_indices].copy()])
        self.values = np.concatenate([self.values[local_indices], external_buffer.values[external_indices].copy()])
        self.log_probs = np.concatenate(
                [self.log_probs[local_indices], external_buffer.log_probs[external_indices].copy()])
        self.returns = np.concatenate([self.returns[local_indices], external_buffer.returns[external_indices].copy()])
        self.pos = num_of_elements(external_indices) + num_of_elements(local_indices)

    def get(self, batch_size: Optional[int] = None, randomize=True) -> Generator[DictRolloutBufferSamples, None, None]:
        if randomize:
            indices = np.random.permutation(self.pos)
        else:
            indices = np.arange(self.pos)
        self.set_indices_and_generator()

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.pos

        start_idx = 0
        if 0 == self.pos:
            yield None
        else:
            while start_idx < self.pos:
                yield self._get_samples(indices[start_idx: start_idx + batch_size])
                start_idx += batch_size
