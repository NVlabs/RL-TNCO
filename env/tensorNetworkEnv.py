import logging
import pickle
from collections import defaultdict

import numpy as np
import opt_einsum as oe
from gym.spaces import Box, Dict

from agents.vec_env import make_vec_env, SubprocVecEnvModified
from env.baseEnv import TensorNetworkEnv, build_networkx_graph, tensors_to_eq, \
	get_tensor_size
from env.baselineEnv import FollowThePathEnv
from utils.main_utils import read_data_file

GREEDY_IND = 0
COST_IND = -1  # without the feasible contents


def get_simulator(config):
	train_files, _ = read_data_file(config)
	train_eqs = []
	baseline_solutions = defaultdict(list)
	if train_files is None:
		eqs, baseline_solutions = None, None
	else:
		if isinstance(train_files, str):
			train_files = [train_files]
		for train_file in train_files:
			with open(train_file, 'rb') as f:
				eqs, baseline_sol, _ = pickle.load(f)
			train_eqs += eqs
			baseline_solutions = {k: baseline_solutions[k] + v for k, v in baseline_sol.items()}
	env_fn = lambda **x: TNLearningEnv(config, eqs=train_eqs, baseline_solutions=baseline_solutions, **x)
	if config['train_params']['dummy_parallelism']:
		env = make_vec_env(env_id=env_fn, n_envs=config['train_params']['n_envs'], seed=0)
	else:
		env = make_vec_env(env_id=env_fn, n_envs=config['train_params']['n_envs'], seed=0,
		                   vec_env_cls=SubprocVecEnvModified)
	env.set_attr(attr_name='block_decreasing_actions', value=config['train_params']['block_decreasing_action_train'])
	return env


def get_robust_features(normalized_slice, median, clip_bound, fn):
	other_end = fn(normalized_slice)
	robust_scale = (other_end - median).astype(np.float64)
	special_treatment_indices = np.isclose(robust_scale, 0)
	if np.any(special_treatment_indices):
		robust_scale[special_treatment_indices] = np.mean(robust_scale[special_treatment_indices])
		special_treatment_indices = np.isclose(robust_scale, 0)
		robust_scale[special_treatment_indices] = 1.0

	robust_feature = (normalized_slice - median) / robust_scale
	robust_feature = np.clip(robust_feature, -clip_bound, clip_bound)
	return robust_feature


class TNLearningEnv(TensorNetworkEnv):
	def __init__(self, config, **kwargs):
		super().__init__(config, **kwargs)
		self.raw_node_features = config['representation']['node_features']
		self.node_features = self.raw_node_features
		self.node_features += 1  # random features column
		self.edge_features = config['representation']['edge_features']
		global_features = config['representation']['global_features']
		self.add_greedy_score = config['representation']['add_greedy_score']
		self.block_decreasing_actions = config['train_params']['block_decreasing_action']
		if not self.use_flops_metric:
			self.edge_features += 2
		if self.add_greedy_score:
			self.edge_features += 1
		self.add_feasible_actions_as_features = self.block_decreasing_actions and self.eqs and len(self.eqs) == 1 and \
		                                        config['train_params']['add_feasible_actions_as_features']
		self.edge_features += 1 if self.add_feasible_actions_as_features else 0
		self.normalizing_indices = np.ones(self.edge_features, dtype=bool)
		if self.config['train_params']['clip_reward'] > 0:
			self.clip_reward = self.config['train_params']['clip_reward']
		if self.add_feasible_actions_as_features:
			self.normalizing_indices[-1] = False
		if self.config['train_params']['edge_normalization'] == 'robust_features':
			self.robust_bound = 100
			self.edge_features += self.normalizing_indices.sum() * 2
			self.fn = [lambda x: np.max(x, axis=0), lambda x: np.min(x, axis=0)]
		self.observation_space = Dict(spaces={
			'X': Box(high=np.inf, low=-np.inf, shape=(self.max_nodes, self.node_features)),
			'edge_index': Box(high=self.max_nodes, low=-1, shape=(2, self.max_edges), dtype=np.compat.long),
			'edge_features': Box(high=np.inf, low=-np.inf, shape=(self.max_edges, self.edge_features)),
			'global_features': Box(high=np.inf, low=-np.inf, shape=(1, global_features)),
			'n_nodes': Box(high=self.max_nodes, low=0, shape=(1,)),
			'n_edges': Box(high=self.max_edges, low=0, shape=(1,)),
			'feasible_actions': Box(high=1, low=0, shape=(self.max_edges,), dtype=bool)
		})
		self.prev_edge_attr_unnormalized = None
		self.best_result = None
		self.episode_info = None
		self.log_scale_greedy_score = self.config['train_params']['log_scale_greedy_score']
		self.single_eq_mode = False
		self.eq_initial_size = None
		self.step_counter = None
		self.size_dict = None
		self.output_tensor = None
		self.suffix_solver = config['train_params'].get('suffix_solver')
		if self.suffix_solver:
			self.auxilary_env = FollowThePathEnv(config)
			self.prefix_length = config['train_params']['max_prefix']

	def reset(self):
		self.new_episode = True
		if self.data:
			logging.info('Entering obsolete code!')
			self.tensors, size_dict = self.data['tensors'], self.data['size_dict']
			self.output_tensor = ''
			self.data = None
		else:
			if self.use_fixed_eqs_set:
				eq, shapes, size_dict = self.eqs[self.counter]
				self.single_eq_mode = len(self.eqs) == 1
			else:
				eq, shapes, size_dict = self.get_eq_shapes_and_sizes()
			if self.debug:
				print(shapes)
			index_per_node, self.output_tensor = self.split_eq_to_tensors(eq)
			self.tensors = [tuple(x) for x in index_per_node]
			self.eq_initial_size = len(self.tensors)
		self.size_dict = size_dict
		self.state, self.network, _, _ = self._get_state(self.tensors, total_return=0)
		self.episode_history = []
		self.episode_info = {'history_by_indices': [],
		                     'history_by_tensors': [],
		                     'baseline_policy': False,
		                     'total_reward': 0}
		self.step_counter = 0
		return self.state

	def update_scale(self, scale) -> None:
		self.reward_normalization_factor = scale

	def update_best_result(self, solution):
		self.best_result = solution

	def get_best_result(self, ):
		return self.best_result

	def _get_state(self, tensors, u=None, v=None, total_return=None):
		edge_list, edge_attr, outer_edge_indicator = self._get_edge_features(tensors, self.size_dict, u, v)
		edge_attr = edge_attr.astype(np.float64)
		self.prev_edge_attr_unnormalized = edge_attr.copy()
		# Path pruning features
		if self.block_decreasing_actions:
			if self.best_result is not None:
				feasible_actions = -edge_attr[:, COST_IND] + total_return >= self.best_result['total_reward']
			else:
				feasible_actions = np.ones(edge_attr.shape[0], dtype=bool)
			min_cost = edge_attr[:, COST_IND].min()
		else:
			feasible_actions = np.ones(edge_attr.shape[0], dtype=bool)
			min_cost = None
		if self.add_feasible_actions_as_features:
			edge_attr = np.concatenate([edge_attr, feasible_actions[:, np.newaxis]], axis=1)

		# Path pruning features
		if self.add_greedy_score and self.log_scale_greedy_score:
			# convert greedy score to sign(greedy score) * log(greedy score)
			# greedy score is at 0
			edge_attr[np.isclose(edge_attr[:, GREEDY_IND], 0), GREEDY_IND] = 1
			edge_attr[:, GREEDY_IND] = np.sign(edge_attr[:, GREEDY_IND]) * np.log(np.abs(edge_attr[:, GREEDY_IND]))
		elif self.add_greedy_score and not self.config['train_params']['edge_normalization'] == 'robust_features':
			# greedy score is at 0
			edge_attr[:, GREEDY_IND] -= edge_attr[:, GREEDY_IND].min()
		if self.config['train_params']['edge_normalization'] == 'column_max':
			col_max = np.max(np.abs(edge_attr[:, self.normalizing_indices]), axis=0)
			col_max = np.maximum(col_max, 1e-5)
			edge_attr[:, self.normalizing_indices] = edge_attr[:, self.normalizing_indices] / col_max
		elif self.config['train_params']['edge_normalization'] == 'robust_features':
			normalized_slice = edge_attr[:, self.normalizing_indices]
			normalized_slice -= normalized_slice.min(axis=0)
			median = np.median(normalized_slice, axis=0)
			robust_features = np.concatenate(
				[get_robust_features(normalized_slice, median, clip_bound=self.robust_bound, fn=self.fn[0]),
				 get_robust_features(normalized_slice, median, clip_bound=self.robust_bound, fn=self.fn[1]),
				 np.log(1 + normalized_slice)], axis=1)
			edge_attr = np.concatenate([robust_features, edge_attr[:, np.logical_not(self.normalizing_indices)]],
			                           axis=1)
		else:
			raise NotImplemented

		network = build_networkx_graph(edge_list, n_nodes=len(tensors))
		topological_features = np.zeros([network.number_of_nodes(), 1], dtype=np.float32)
		n_nodes = network.number_of_nodes()
		n_edges = network.number_of_edges()
		assert n_edges <= self.max_edges and n_nodes <= self.max_nodes, "Network bounds are too random_TNs"

		# padding and building up the state
		edge_list = np.pad(edge_list, pad_width=[(0, 0,), (0, self.max_edges - n_edges)])
		edge_attr = np.pad(edge_attr, pad_width=[(0, self.max_edges - n_edges), (0, 0)]).astype(np.float32)
		feasible_actions = np.pad(feasible_actions, pad_width=[(0, self.max_edges - n_edges)])
		X = topological_features
		X = np.pad(X, pad_width=[(0, self.max_nodes - n_nodes), (0, 0)]).astype(np.float32)
		state = {'X': X,
		         'edge_index': edge_list,
		         'edge_features': edge_attr,
		         'global_features': np.ones([1, 2], dtype=np.float32),  # None at this point
		         'n_nodes': np.array([n_nodes]),
		         'n_edges': np.array([n_edges]),
		         'feasible_actions': feasible_actions}
		return state, network, feasible_actions, min_cost

	def _update_edge_features_after_contraction(self, u, v, previous_network, tensors, size_dict):
		"""
		This function updates the edge_list and the edge_attr based on the previous edge_list and edge_attr. It updates only
		thed edged impinging on either u and v.
		"""
		edge_list = list()
		edge_attr = list()
		relevant_slice = self.state['edge_index'][:, :self.state['n_edges'][0]]
		row_indices = np.logical_or(relevant_slice == u, relevant_slice == v)
		row_indices = np.logical_not(np.logical_or(row_indices[0], row_indices[1]))
		remaining_slice = relevant_slice[:, row_indices]
		remaining_slice[remaining_slice > v] -= 1
		remaining_slice[remaining_slice > u] -= 1
		new_tensor_neighbors_indices = set(previous_network.neighbors(u)).union(previous_network.neighbors(v)) - {u, v}
		# find edges
		new_neighbors = [set(tensors[i]) for i in new_tensor_neighbors_indices]
		tensor_sizes = [get_tensor_size(x, size_dict) for x in new_neighbors]
		seti = set(tensors[-1])
		new_tensor_size = get_tensor_size(tensors[-1], size_dict)
		i = len(tensors) - 3
		for j, setj, candidate_size in zip(new_tensor_neighbors_indices, new_neighbors, tensor_sizes):
			intersection = seti.intersection(setj)
			index_union = seti.union(setj)
			common_weight = get_tensor_size(intersection, size_dict)
			if common_weight > 1:
				ind = j - 1 if j > v else j
				ind -= 1 if j > u else 0
				edge_list.append(sorted([i, ind]))
				edge_attr_tensor = self._get_edge_attribute(common_weight, new_tensor_size, candidate_size, index_union,
				                                            size_dict, intersection)
				edge_attr.append(edge_attr_tensor)
		if len(edge_list) > 0:
			edge_list = np.array(edge_list, dtype=np.int64).T
			edge_list = np.concatenate([remaining_slice, edge_list], axis=1)
			edge_attr = np.array(edge_attr)
			edge_attr = np.concatenate([self.prev_edge_attr_unnormalized[row_indices], edge_attr])
		else:
			edge_list = remaining_slice
			edge_attr = self.prev_edge_attr_unnormalized[row_indices]
		return edge_list, edge_attr

	def _get_edge_features_from_scratch(self, tensors, size_dict):
		edge_list = list()
		edge_attr = list()
		# find edges
		tensor_set = [set(x) for x in tensors]
		tensor_sizes = [get_tensor_size(x, size_dict) for x in tensors]
		for i in range(len(tensors)):
			seti = tensor_set[i]
			for j in range(i + 1, len(tensors)):
				setj = tensor_set[j]
				intersection = seti.intersection(setj)
				index_union = seti.union(setj)
				common_weight = get_tensor_size(intersection, size_dict)
				if common_weight > 1:
					edge_list.append((i, j))
					edge_attr_tensor = self._get_edge_attribute(common_weight, tensor_sizes[i], tensor_sizes[j],
					                                            index_union,
					                                            size_dict, intersection)
					edge_attr.append(edge_attr_tensor)
		edge_list = np.array(edge_list, dtype=np.int64).T
		edge_attr = np.array(edge_attr, dtype=np.float64)
		return edge_list, edge_attr

	def _get_edge_attribute(self, common_weight, tensor_size_a, tensor_size_b, index_union, size_dict, intersection):
		# common weight - the span of the index represented by the edge
		flop_weight = get_tensor_size(index_union, size_dict)
		if self.add_greedy_score:
			greedy_score = tensor_size_a + tensor_size_b - get_tensor_size(index_union - intersection,
			                                                               size_dict=size_dict)
			edge_attr_tensor = (greedy_score, common_weight, flop_weight)
		else:
			edge_attr_tensor = (common_weight, flop_weight)
		return edge_attr_tensor

	def _get_edge_features(self, tensors, size_dict, u, v):
		build_from_scratch = u is None
		if build_from_scratch:
			edge_list, edge_attr = self._get_edge_features_from_scratch(tensors, size_dict)
		else:
			edge_list, edge_attr = self._update_edge_features_after_contraction(u, v, self.network, tensors, size_dict)
			tensors.pop(v)
			tensors.pop(u)
		return edge_list, edge_attr, None

	def step(self, action):
		u, v = self.state['edge_index'][:, action]
		u, v = int(min(u, v)), int(max(u, v))
		new_tensor = tuple(set(self.tensors[u]).symmetric_difference(set(self.tensors[v])))
		reward, rewards_info = self.get_reward(u, v, new_tensor)
		self.episode_info['total_reward'] += reward
		self.episode_info['history_by_indices'].append((u, v))
		self.episode_info['history_by_tensors'].append((self.tensors[u], self.tensors[v]))
		self.episode_history.append((u, v))
		reward = reward / self.reward_normalization_factor
		previous_network = self.network
		done = previous_network.number_of_edges() == 1
		if not done:
			# we add a tensor if the contraction did not compress to a null graph a disconnected component
			if len(new_tensor) > 0:
				self.tensors.append(new_tensor)
			self.state, self.network, feasible_actions, min_cost = \
				self._get_state(self.tensors, u, v, total_return=self.episode_info['total_reward'])
			penalty = 0
			if not np.any(feasible_actions):
				# A heuristic that estimates the future value of a "pruned" terminal state (a state where the total accumulated
				# cost + any action cost is greater than the minimal cost) based on the best known cost and the current path cost
				done = True
				alpha = 0.5
				mean_best_result_cost = self.best_result['total_reward'] / self.eq_initial_size
				mean_current_path_cost = self.episode_info['total_reward'] / (
						self.eq_initial_size - len(self.tensors) + 1e-8)
				future_term = 2 * (len(self.tensors) - 1) \
				              * (mean_best_result_cost * alpha + mean_current_path_cost * (1 - alpha))
				penalty = - abs(min_cost) + future_term

			if self.suffix_solver and self.step_counter == self.prefix_length:
				# using a predefined solver, apply its on the first k steps

				# define an equation for the solver
				eq, shapes = tensors_to_eq(self.tensors, self.output_tensor, self.size_dict)
				path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=self.suffix_solver)
				self.auxilary_env.reset(self.tensors, eq, shapes, self.size_dict, path)
				auxilary_env_done = False

				# follow the solver path
				while not auxilary_env_done:
					_, aux_step_reward, auxilary_env_done, _ = self.auxilary_env.step()
					penalty += aux_step_reward
				done = True
			if penalty:
				reward += penalty / self.reward_normalization_factor
				self.episode_info['total_reward'] += penalty
				rewards_info['flops_only'] -= penalty

		if done:
			if self.eqs:
				self.counter = (self.counter + 1) % len(self.eqs)
			self.network = None
			self.state = {'X': np.zeros([self.node_features, self.max_nodes], dtype=np.float32),
			              'edge_index': np.zeros([2, self.max_edges]),
			              'edge_features': np.zeros([self.max_edges, self.edge_features], dtype=np.float32),
			              'global_features': np.ones([2], dtype=np.float32),  # None at this point
			              'n_nodes': 1,
			              'n_edges': 1}
			if self.single_eq_mode and \
					(self.best_result is None or self.episode_info['total_reward'] > self.best_result['total_reward']):
				self.update_best_result(self.episode_info)
		self.new_episode = False
		if self.clip_reward > 0 and self.reward_normalization_factor > 1:
			reward = max(-self.clip_reward, reward)
		self.step_counter += 1
		return self.state, reward, done, rewards_info
