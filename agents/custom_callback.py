import logging
import os
import pickle
import time
from abc import ABC
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from agents.vec_env import DummyVecEnvModified as DummyVecEnv
from agents.vec_env import SubprocVecEnvModified as SubprocVecEnv
from agents.vec_env import make_vec_env
from env.baselineEnv import BaselineEnv
from env.tensorNetworkEnv import TNLearningEnv
from utils.evaluation_utils import maybe_update_best_result, get_best_result
from utils.general_usage_utils import get_torch_rng_states
from utils.main_utils import read_data_file


class CuQuantumCallback(BaseCallback, ABC):
    """
    The CuQuantum callback, which added extra functions to the BaseCallback class
    """

    def __init__(self, config, verbose=0):
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        super(CuQuantumCallback, self).__init__(verbose)
        self.reward_normalization_factor = None
        self.config = config

        # load the data files for training and evaluation (if needed)
        train_files, eval_file = read_data_file(config)
        assert eval_file, 'Must equip with test set'
        with open(eval_file, 'rb') as f:
            eqs, baseline_solutions, _ = pickle.load(f)

        # set the number of enviroments per equation and assign enviroments to equations
        max_envs = min(config['eval']['max_eval_envs'], len(eqs) * config['eval']['long_eval_episodes'])
        self.n_simultaneous_path_per_eq = max_envs // len(eqs)
        n_long_test_envs = self.n_simultaneous_path_per_eq * len(eqs)
        self.n_long_eval_repeats = int(config['eval']['long_eval_episodes'] / self.n_simultaneous_path_per_eq)
        self.short_eval_same_eq_subsets = [[i] for i in range(len(eqs))]
        self.long_eval_same_eq_subsets = [np.arange(i, n_long_test_envs, len(eqs)) for i in range(len(eqs))]
        if config['train_params']['dummy_parallelism']:
            short_vec_env_cls = DummyVecEnv
            long_vec_env_cls = DummyVecEnv
        else:
            short_vec_env_cls = SubprocVecEnv if 2 < len(eqs) <= 50 else DummyVecEnv
            long_vec_env_cls = SubprocVecEnv if 2 < n_long_test_envs <= 50 else DummyVecEnv

        env_fn = lambda **x: TNLearningEnv(config, eqs=eqs, **x)
        self.long_test_env = make_vec_env(env_id=env_fn, n_envs=n_long_test_envs, seed=config['eval']['eval_seed'],
                                          env_kwargs={'evaluation_env': True}, vec_env_cls=long_vec_env_cls)
        self.short_test_env = make_vec_env(env_id=env_fn, n_envs=len(eqs), seed=config['eval']['eval_seed'],
                                           env_kwargs={'evaluation_env': False}, vec_env_cls=short_vec_env_cls)
        train_eqs = []
        self.counter = 0
        self.n_fixed_train_eqs = None
        self.scaling_env = self.short_test_env
        self.train_same_eq_subsets = None
        self.episode_total_return = None
        self.callback_time = None
        # used for running-average of the log
        self.total_return_momentum = self.config['train_params']['total_return_momentum']
        self.best_gap_to_optimality = {}
        if train_files:
            # creates a scaling env based on the first max_env_for_scaling eqs from the train set
            max_env_for_scaling = 10
            train_baseline_solutions = defaultdict(list)
            if isinstance(train_files, str):
                train_files = [train_files]
            for train_file in train_files:
                with open(train_file, 'rb') as f:
                    train_eq, baseline_sol, _ = pickle.load(f)
                train_eqs += train_eq
                train_baseline_solutions = {k: train_baseline_solutions[k] + v for k, v in baseline_sol.items()}
            self.n_fixed_train_eqs = len(train_eqs)
            if config['train_params']['n_envs'] >= self.n_fixed_train_eqs:
                self.train_same_eq_subsets = [np.arange(i, config['train_params']['n_envs'], self.n_fixed_train_eqs)
                                              for i in range(self.n_fixed_train_eqs)]
            else:
                self.train_same_eq_subsets = None
            self.scaling_env = make_vec_env(
                    env_id=lambda **x: TNLearningEnv(config, eqs=train_eqs[:max_env_for_scaling], evaluation_env=False,
                                                     **x),
                    n_envs=max_env_for_scaling, seed=config['eval']['eval_seed'], )

        # creates a baseline env and log its metrics
        baseline_env = make_vec_env(
                env_id=lambda **x: BaselineEnv(config, solver=self.config['train_params']['precalc_solver'],
                                               evaluation_env=True, eqs=eqs, baseline_solutions=baseline_solutions, **x),
                n_envs=len(eqs), seed=config['eval']['eval_seed'])
        self.baseline_metrics = self._get_metrics(env=baseline_env, use_model_actions=False)
        if self.config['train_params']['baseline_for_action_elimination_in_eval']:
            baseline_env = make_vec_env(
                    env_id=lambda **x: BaselineEnv(config,
                                                   solver=self.config['train_params'][
                                                       'baseline_for_action_elimination_in_eval'],
                                                   evaluation_env=True, eqs=eqs, baseline_solutions=baseline_solutions,
                                                   **x),
                    n_envs=len(eqs), seed=config['eval']['eval_seed'])
            self.action_elimination_baseline_metrics_eval = self._get_metrics(env=baseline_env, use_model_actions=False)
        if self.config['train_params']['baseline_for_action_elimination_in_train'] and \
                self.config['train_params']['block_decreasing_action_train'] and len(train_eqs) > 0:
            baseline_env = make_vec_env(
                    env_id=lambda **x: BaselineEnv(config,
                                                   solver=self.config['train_params'][
                                                       'baseline_for_action_elimination_in_train'],
                                                   #
                                                   # fixed_eqs_set_size=config['eval']['eval_episodes'],
                                                   evaluation_env=True, eqs=train_eqs,
                                                   baseline_solutions=train_baseline_solutions, **x),
                    n_envs=len(train_eqs), seed=config['eval']['eval_seed'])
            self.action_elimination_baseline_metrics_train = self._get_metrics(env=baseline_env,
                                                                               use_model_actions=False)
        else:
            self.action_elimination_baseline_metrics_train = None

        n_iterations = 5000
        self.results = {'baseline': self.baseline_metrics['episode_rewards'].flatten(),
                        'ours'    : np.zeros([n_iterations, len(eqs)])}

    def _get_metrics(self, env, use_model_actions=True, deterministic=True):
        """
        For each env, counts the total cost of the trajectory generated by the model
        """
        actions = np.zeros(env.num_envs)
        acc_ep_reward = np.zeros(env.num_envs, dtype=float)
        flops_only = np.zeros(env.num_envs, dtype=float)
        roofline_model = np.zeros(env.num_envs, dtype=float)
        completed_pass = np.zeros(env.num_envs, dtype=bool)
        epidose_length = np.zeros(env.num_envs, dtype=float)
        episode_rewards = np.zeros(env.num_envs, dtype=float)
        observations = env.reset()
        counter = 0
        metrics = {}
        while not np.all(completed_pass):
            if use_model_actions:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(observations, self.model.device)
                    actions, values, _ = self.model.policy.forward(obs_tensor, deterministic=deterministic)
                    if counter == 0:
                        metrics['value_at_0'] = values.cpu().numpy().reshape(-1, 1)
            observations, reward, done, info = env.step(actions)
            newly_finished = np.logical_and(done, np.logical_not(completed_pass))
            acc_ep_reward += reward
            epidose_length[newly_finished] = counter
            episode_rewards[newly_finished] = acc_ep_reward[newly_finished]
            # count only if this is still the first episode, i.e., pass was not completed yet
            flops_only += np.array([x['flops_only'] for x in info], dtype=float) * np.logical_not(completed_pass)
            roofline_model += np.array([x['total_time'] for x in info], dtype=float) * np.logical_not(completed_pass)
            completed_pass = np.logical_or(completed_pass, done)
            counter += 1
        metrics.update({'flops_only'     : flops_only.reshape(-1, 1),
                        'roofline_model' : roofline_model.reshape(-1, 1),
                        'episode_rewards': episode_rewards.reshape(-1, 1)})
        return metrics

    def evaluate_model(self):
        do_long_evaluation = self.counter % (self.config['eval']['long_eval_freq']) == 0 and self.counter > 0
        if self.counter % self.config['eval']['eval_freq'] == 0:
            logging.info(f'Start evaluation, epoch {self.counter}')
            if do_long_evaluation:
                logging.info('==========   Starting long evaluation  ============')
            self._evaluate_episode(do_long_evaluation=do_long_evaluation)
            logging.info(f'Finished evaluation, epoch {self.counter}')

    def _evaluate_episode(self, env=None, deterministic=True, log=True, do_long_evaluation=False):
        n_paths = self.n_simultaneous_path_per_eq
        if env is None:
            env = self.short_test_env
        reward_normalization_factor = self.model.env.get_attr('reward_normalization_factor')[0]
        previous_best_result = get_best_result(self.short_test_env, same_eq_subsets=self.short_eval_same_eq_subsets)
        start_time = time.time()
        metrics = self._get_metrics(env, use_model_actions=True, deterministic=deterministic)
        logging.info(f'Mean time per episode in short evaluation: {(time.time() - start_time) / env.num_envs}')
        if do_long_evaluation:
            start_time = time.time()
            for n in range(self.n_long_eval_repeats):
                long_eval_metrics = self._get_metrics(self.long_test_env, use_model_actions=True, deterministic=False)
                if self.n_simultaneous_path_per_eq > 1:
                    best_long_eval_score = get_best_result(self.long_test_env,
                                                           same_eq_subsets=self.long_eval_same_eq_subsets)
                    for j, subset in enumerate(self.long_eval_same_eq_subsets):
                        self.long_test_env.set_attr(attr_name='best_result', value=best_long_eval_score[j],
                                                    indices=subset)
                for key, value in metrics.items():
                    reshaped_values = long_eval_metrics[key].reshape(-1, self.n_simultaneous_path_per_eq)
                    metrics[key] = np.min(np.abs(np.concatenate([np.abs(value), reshaped_values], axis=1)), axis=1,
                                          keepdims=True)
            logging.info(f'Time for {self.n_long_eval_repeats} repeats per episode in long evaluation: '
                         f'{n_paths * (time.time() - start_time) / self.long_test_env.num_envs}')
            logging.info(f'Evaluation time: {time.time() - start_time}')
        episode_rewards = metrics['episode_rewards'].reshape(-1, 1)
        episode_rewards = np.abs(episode_rewards)
        episode_rewards = episode_rewards * reward_normalization_factor
        unique_values = pd.DataFrame(episode_rewards.transpose()).nunique().to_numpy()
        episode_rewards = episode_rewards.min(axis=1, keepdims=True)
        mean_reward = np.mean(episode_rewards)
        if log:
            if not self.config['eval']['reset_eval_env_after_evaluation']:
                self.update_best_result()
            self.logger.record(key="evaluation/reward_normalization_factor",
                               value=reward_normalization_factor)
            value_at_0 = metrics['value_at_0'] * reward_normalization_factor
            if self.config['train_params']['reward_function'] == 'flops_only':
                mean_baseline_cost, mean_reward = self.log_performance_metrics(metrics['flops_only'],
                                                                               self.baseline_metrics['flops_only'],
                                                                               'flops_only')
                self.log_performance_metrics(metrics['roofline_model'], self.baseline_metrics['roofline_model'],
                                             'roofline_model')
            else:
                self.log_performance_metrics(metrics['flops_only'], self.baseline_metrics['flops_only'], 'flops_only')
                mean_baseline_cost, mean_reward = self.log_performance_metrics(metrics['roofline_model'],
                                                                               self.baseline_metrics['roofline_model'],
                                                                               'roofline_model')
            estimated_value_to_optimality_gap = \
                (value_at_0 - self.baseline_metrics['episode_rewards']).mean() / mean_baseline_cost
            estimated_value_to_optimality_gap_abs = \
                np.abs(value_at_0 - self.baseline_metrics['episode_rewards']).mean() / mean_baseline_cost
            if self.model.best_results is not None and self.model.best_results['eval'] is not None and \
                    not self.config['eval']['reset_eval_env_after_evaluation']:  # logic is flawed otherwise
                best_recorded_results = self.model.best_results['eval']
                if not (isinstance(best_recorded_results, list)):
                    best_recorded_results = [best_recorded_results]
                best_results = np.array([x['total_reward'] for x in best_recorded_results])
                best_result_to_baseline_gap = (abs(best_results) - mean_baseline_cost) / mean_baseline_cost
                self.logger.record("evaluation/best_result_to_baseline_gap",
                                   float(np.mean(best_result_to_baseline_gap)))
            self.logger.record("evaluation/estimated_value_to_baseline_gap", float(estimated_value_to_optimality_gap))
            self.logger.record("evaluation/mean_number_of_unique_paths", float(np.mean(unique_values)))
            self.logger.record("evaluation/estimated_value_to_baseline_gap_abs",
                               float(estimated_value_to_optimality_gap_abs))
            if len(episode_rewards) > 1:
                wandb.log({"evaluation/rewards_histogram": wandb.Histogram(np.abs(episode_rewards)),
                           "evaluation/optimal_histogram": wandb.Histogram(
                                   np.abs(self.baseline_metrics['episode_rewards']))})
            if self.config['eval']['reset_eval_env_after_evaluation']:
                self.sync_solution_in_evaluation_envs(previous_best_result)
            self.results['ours'][self.counter] = episode_rewards.T
        return mean_reward

    def log_performance_metrics(self, episode_rewards, baseline_rewards, key):
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        self.logger.record(f"evaluation/{key}_mean_reward", mean_reward)
        self.logger.record(f"evaluation/{key}_std_reward", std_reward)
        mean_baseline_cost = np.abs(np.mean(baseline_rewards))
        std_baseline_cost = np.abs(np.std(baseline_rewards))
        self.logger.record(f"evaluation/{key}_mean_baseline_cost", mean_baseline_cost)
        self.logger.record(f"evaluation/{key}_std_baseline_cost", std_baseline_cost)
        gap_to_optimality = (abs(mean_reward) - mean_baseline_cost) / mean_baseline_cost
        self.best_gap_to_optimality[key] = min(gap_to_optimality, self.best_gap_to_optimality[key]) \
            if key in self.best_gap_to_optimality else gap_to_optimality
        self.logger.record(f"evaluation/{key}_gap_to_optimality", gap_to_optimality)
        self.logger.record(f"evaluation/{key}_best_gap_to_optimality", self.best_gap_to_optimality[key])
        self.logger.record(f"evaluation/{key}_best_gain", 1 / (self.best_gap_to_optimality[key] + 1))
        return mean_baseline_cost, mean_reward

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any], external_constant=None) -> None:
        # Those are reference and will be updated automatically
        if locals_ is not None:
            self.locals = locals_
        if globals_ is not None:
            self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start(external_constant=external_constant)

    def _on_training_start(self, external_constant=None) -> None:
        """
        This method is called before the first rollout starts.
        """

        if self.config['train_params']['limit_to_baseline_improving_actions']:
            # send the baseline solution to the right environment
            baseline_rewards = self.action_elimination_baseline_metrics_eval['episode_rewards'].flatten()
            baseline_solution = [{'baseline_policy': True, 'total_reward': x} for x in baseline_rewards]
            self.sync_solution_in_evaluation_envs(baseline_solution)
            if self.config['train_params']['sync_mode'] == 'all':
                # If all environments share the same equation, feed the baseline solution to all environments
                self.model.env.set_attr(attr_name='best_result', value=baseline_solution[0])
            elif self.action_elimination_baseline_metrics_train is not None:
                # If different environments have different equations, feed the corresponding solution to each
                # environment
                baseline_rewards = self.action_elimination_baseline_metrics_train['episode_rewards'].flatten()
                baseline_solution = [{'baseline_policy': True, 'total_reward': x} for x in baseline_rewards]
                for j, subset in enumerate(self.train_same_eq_subsets):
                    self.model.env.set_attr(attr_name='best_result', value=baseline_solution[j], indices=subset)
            else:
                logging.info('Train environments were not synced.')

        self.update_reward_normalization_factor(external_constant=external_constant)

    def sync_solution_in_evaluation_envs(self, ref_result):
        if isinstance(ref_result, list) or isinstance(ref_result, (np.ndarray, np.generic)):
            for j in range(self.short_test_env.num_envs):
                self.short_test_env.set_attr(attr_name='best_result', value=ref_result[j], indices=[j])
            for j, subset in enumerate(self.long_eval_same_eq_subsets):
                self.long_test_env.set_attr(attr_name='best_result', value=ref_result[j], indices=subset)
        else:  # baseline_solution is dictionary
            self.long_test_env.set_attr(attr_name='best_result', value=ref_result)
            self.short_test_env.set_attr(attr_name='best_result', value=ref_result)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        t = time.time()
        if self.counter % self.config['train_params']['save_freq'] == 0 and self.counter > 0:
            self.evaluate_model()
            self.model.reset_info_buffer()
            self.update_best_result()
            self.save_model_and_stats()
        self.counter += 1
        self.callback_time = time.time() - t

    def save_model_and_stats(self):
        logging.info(f'Saving model, epoch {self.counter}')
        filename = os.path.join(wandb.run.dir, f'epoch_{self.counter}.model')
        self.model.save(filename)
        wandb.save(filename)
        filename = os.path.join(wandb.run.dir, f'stats_{self.counter}.model')
        with open(filename, 'wb') as f:
            data_to_save = {'solution'                   : self.model.best_results,
                            'reward_normalization_factor': self.reward_normalization_factor,
                            'rewards_per_episode'        : self.results,
                            'numpy_rng_state'            : np.random.get_state(),
                            'torch_rng_state'            : get_torch_rng_states(self.model.device)}
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        wandb.save(filename)

    def update_best_result(self):
        best_result = None
        best_training_score = get_best_result(self.model.env, same_eq_subsets=self.train_same_eq_subsets)
        best_short_eval_score = get_best_result(self.short_test_env, same_eq_subsets=self.short_eval_same_eq_subsets)
        best_long_eval_score = get_best_result(self.long_test_env, same_eq_subsets=self.long_eval_same_eq_subsets)
        self.model.best_results = {'train'    : best_training_score,
                                   'eval'     : best_short_eval_score,
                                   'long_eval': best_long_eval_score}
        if self.config['train_params']['sync_mode'] == 'train' and self.train_same_eq_subsets:
            for j, subset in enumerate(self.train_same_eq_subsets):
                self.model.env.set_attr(attr_name='best_result', value=best_training_score[j], indices=subset)
        # 'all_but_train should be depracated
        elif self.config['train_params']['sync_mode'] in ['all', 'all_but_train', 'eval_envs']:
            best_scaling_env_score = get_best_result(self.scaling_env)
            if self.config['train_params']['sync_mode'] == 'all':
                scores = [best_training_score[0], best_short_eval_score[0], best_long_eval_score[0],
                          best_scaling_env_score]
            elif self.config['train_params']['sync_mode'] == 'eval_envs':
                scores = [best_short_eval_score, best_long_eval_score]
            else:
                scores = [best_short_eval_score, best_long_eval_score, best_scaling_env_score]
            for candidate in scores:
                best_result = maybe_update_best_result(best_result, candidate)
            if best_result is not None:
                if isinstance(best_result, dict):
                    assert self.config['train_params']['sync_mode'] == 'all'
                    self.update_best_result_in_envs(best_result, update_scaling_env=True,
                                                    update_train_env=self.config['train_params']['sync_mode'] == 'all')
                if isinstance(best_result, list):
                    assert self.config['train_params']['sync_mode'] == 'eval_envs'
                    self.sync_solution_in_evaluation_envs(best_result)
            self.model.best_results['eval'] = best_result
        return best_result

    def update_best_result_in_envs(self, best_result, update_scaling_env=True, update_train_env=True, indices=None):
        self.short_test_env.set_attr(attr_name='best_result', value=best_result, indices=indices)
        self.long_test_env.set_attr(attr_name='best_result', value=best_result, indices=indices)
        if update_scaling_env:
            self.scaling_env.set_attr(attr_name='best_result', value=best_result, indices=indices)
        if update_train_env:
            self.model.env.set_attr(attr_name='best_result', value=best_result, indices=indices)

    def update_reward_normalization_factor(self, external_constant=None):
        if external_constant:
            logging.info(f'=======Loading reward normalization =======')
            reward_normalization_factor = external_constant
        else:
            logging.info(f'=======Calculating initial reward normalization =======')
            reward_normalization_factor = - self._evaluate_episode(env=self.scaling_env, log=False, deterministic=False)
            logging.info(f'Scaling env resulted in mean reward of {reward_normalization_factor:.4g}')
            if abs(reward_normalization_factor) > 1e20:
                logging.warning(
                        'Reducing reward_normalization_factor to 1e20. Reward normalization by random trajectory failed')
                reward_normalization_factor = self.config['train_params']['external_reward_normalization']
            else:
                reward_normalization_factor = \
                    abs(reward_normalization_factor * self.config['train_params']['additional_rescaling_factor'])
        logging.info(f'Setting Reward normalization factor to: {reward_normalization_factor:.4g}')
        self.long_test_env.set_attr('reward_normalization_factor', reward_normalization_factor)
        self.short_test_env.set_attr('reward_normalization_factor', reward_normalization_factor)
        self.model.env.set_attr('reward_normalization_factor', reward_normalization_factor)
        self.reward_normalization_factor = reward_normalization_factor

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        episode_total_return, episode_median_return, mean_episode_evaluated_return = \
            get_mean_episode_return(self.model.rollout_buffer)
        out_string = str()
        if self.episode_total_return is None:
            self.episode_total_return = episode_total_return
        elif self.model.episode_total_return is not None:
            self.episode_total_return = self.episode_total_return * self.total_return_momentum + \
                                        episode_total_return * (1 - self.total_return_momentum)
        if self.episode_total_return is not None and episode_median_return is not None:
            out_string += f' episode_total_return: {episode_total_return:5g}, episode_median_return: ' \
                          f'{episode_median_return:5g}, acc_episode_total_return: {self.episode_total_return:5g}'
            info_summary = dict()
            for k in self.model.info_buffer[0].keys():
                if k in ['terminal_observation', 'episode', 'cost_at_0']:
                    continue
                info_summary[k] = np.array([x[k] for x in self.model.info_buffer], dtype=float)
            for k, v in info_summary.items():
                self.logger.record(f"train/{k}", np.nanmean(v))
            if len(self.model.ep_cost_buffer) > 0:
                self.logger.record(f"train/solver cost at 0", np.mean(self.model.ep_cost_buffer))
                self.logger.record(f"train/solver cost at 0 std", np.std(self.model.ep_cost_buffer))
        if mean_episode_evaluated_return is not None:
            out_string += f'estimated_return: {mean_episode_evaluated_return:5g}'
            self.logger.record(f"train/estimated value at 0", np.mean(mean_episode_evaluated_return))
        if len(out_string) > 0:
            logging.info(out_string)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.save_model_and_stats()
        pass

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True


def get_mean_episode_return(rollout_buffer):
    total_returns = []
    return_estimation = []
    completed_episodes = 0
    evaluated_episodes = 0
    for env in range(rollout_buffer.episode_starts.shape[1]):
        indices = np.where(rollout_buffer.episode_starts[:, env])[0][:-1]
        if len(indices) > 0:
            total_returns += rollout_buffer.returns[indices, env].tolist()
            completed_episodes += len(indices)
        else:
            indices = np.where(rollout_buffer.episode_starts[:, env])[0]
            return_estimation += rollout_buffer.values[indices, env].tolist()
            evaluated_episodes += len(indices)
    # mean_episode_total_return = total_returns / compleded_episodes
    if completed_episodes > 0:
        mean_episode_total_return = np.mean(total_returns)
        median_return = np.median(total_returns)
    else:
        mean_episode_total_return, median_return = None, None
    if evaluated_episodes > 0:
        mean_episode_evaluated_return = np.mean(return_estimation)
    else:
        mean_episode_evaluated_return = None
    return mean_episode_total_return, median_return, mean_episode_evaluated_return
