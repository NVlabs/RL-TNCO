import logging
import math
import random
import string

import gym
import networkx as nx
import numpy as np
import opt_einsum as oe
from gym.spaces import Discrete
from numpy.random import default_rng


class TensorNetworkEnv(gym.Env):
	def __init__(self, config, fixed_eqs_set_size=None, debug=False, evaluation_env=False, env_id=None, eqs=None,
	             baseline_solutions=None, partition_eqs_to_workers=True):
		super().__init__()
		self.config = config
		self.env_id = env_id
		# self.seed()
		self.max_nodes = config['network']['n_nodes']
		self.max_edges = int(config['network']['n_nodes'] * (config['network']['n_nodes'] - 1) / 2)
		if self.max_edges > config['network']['n_edges']:
			self.max_edges = config['network']['n_edges']
			# logging.info(f'Warning: Reducing maximal number of edges to {self.max_edges}')
		self.action_space = Discrete(self.max_edges)
		self.reward_normalization_factor = 1.0
		self.use_flops_metric = True
		self.scale = 1
		# self.visualize = visualize if visualize is not None else config['visualization']['visualize']
		self.win, self.fig, self.video_writer = None, None, None
		self.setup_video_writer = True
		self.additional_colormap = None
		self.tensors, self.state, self.network = None, None, None
		self.debug = debug
		self.new_episode = None
		self.evaluation_env = evaluation_env
		self.counter = 0
		self.env_id = env_id
		self.eqs, self.baseline_solutions = None, None
		if eqs is not None and len(eqs) > 0:
			if env_id is not None and partition_eqs_to_workers:
				local_index = env_id % len(eqs)
				local_eqs = [eqs[local_index]]
				if baseline_solutions is not None:
					local_baseline_solutions = {}
					for key, sols in baseline_solutions.items():
						local_baseline_solutions[key] = [baseline_solutions[key][local_index]]
				else:
					local_baseline_solutions = None
			else:
				local_eqs = eqs
				local_baseline_solutions = baseline_solutions
				if env_id is not None:
					self.counter = env_id % len(eqs)
			self.eqs, self.baseline_solutions = local_eqs, local_baseline_solutions
		self.episode_history = None
		self.data = None
		self.scale = 1

		self.use_fixed_eqs_set = False
		if self.eqs:
			self.use_fixed_eqs_set = True
		elif evaluation_env:
			logging.info('Generating fixed set of equations')
			assert fixed_eqs_set_size is not None, 'Number of validation / test equations was not set'
			self.eqs = [self.get_eq_shapes_and_sizes() for _ in range(fixed_eqs_set_size)]
			self.use_fixed_eqs_set = True

	def reset(self):
		raise NotImplementedError

	@staticmethod
	def split_eq_to_tensors(eq):
		if '->' in eq:
			input_tensors, output_tensor = str.split(eq, '->')
		else:
			input_tensors = eq
			output_tensor = ''
		input_tensors = str.split(input_tensors, ',')
		return input_tensors, output_tensor

	def get_eq_shapes_and_sizes(self):
		if self.config['network']['tensor_dim']:
			eq, shapes, size_dict = oe.helpers.rand_equation(n=self.config['network']['n_nodes'],
			                                                 n_out=2, d_min=2,
			                                                 d_max=self.config['network']['tensor_dim'],
			                                                 reg=self.config['network']['mean_connectivity'],
			                                                 return_size_dict=True)
		else:
			eq, shapes, size_dict = oe.helpers.rand_equation(n=self.config['network']['n_nodes'],
			                                                 n_out=2,
			                                                 reg=self.config['network']['mean_connectivity'],
			                                                 return_size_dict=True)
		return eq, shapes, size_dict

	def get_reward(self, u, v, new_tensor):
		"""
		:param u,v : the selected edge in terms of node indices
		:param new_tensor: the resulting tensor after the contraction of the u and v tensors.
		:return: the cost (-reward) of the selected edge contraction
		"""
		# required_flops = edge_contraction_costs[action].cpu().numpy() * self.scaling_factor
		flop_cost = float(get_tensor_size(tuple(set(self.tensors[u]).union(set(self.tensors[v]))), self.size_dict))
		required_flops = flop_cost
		required_memory = get_tensor_size(self.tensors[u], self.size_dict) + get_tensor_size(self.tensors[v],
		                                                                                     self.size_dict) \
		                  + get_tensor_size(new_tensor, self.size_dict)
		required_memory = float(required_memory)
		rewards_info = {'flops_only': required_flops,
		                'total_time': max(required_memory / self.scale, required_flops)}
		if self.config['train_params']['reward_function'] == 'flops_only':
			reward = - rewards_info['flops_only']
		# self.tensors - a list of tuples, each tuple correspond to the indices of node, e.g. ('A','C','D')
		elif self.config['train_params']['reward_function'] == 'total_time':
			reward = - rewards_info['total_time']
		else:
			raise Exception('Unknown config.train_params.reward_function')
		return reward, rewards_info

	def update_scale(self, scale):
		logging.info(f'updated scale in env {self.env_id}: {scale}')
		self.reward_normalization_factor = scale

	def step(self, action):
		return NotImplementedError

	def seed(self, seed=None):
		if seed is None:
			seed = self.config['train_params']['seed']
		random.seed(seed)
		np.random.seed(seed)


def build_networkx_graph(edge_list, n_nodes):
	network = nx.Graph()
	network.add_edges_from(edge_list.T.tolist())
	network.add_nodes_from([i for i in range(n_nodes)])
	return network


def generate_fixed_number_of_edges(n, num_extra_edges=2):
	eq = ''
	size_dict = {}
	chars = string.ascii_uppercase
	for ii in range(n + num_extra_edges):
		size_dict[chars[ii]] = np.random.randint(2, 10)

	rng = default_rng()
	extra_edges_start_nodes = rng.choice(int(n / 2) - 1, size=num_extra_edges, replace=False)
	extra_edges_end_nodes = int(n / 2) + 1 + rng.choice(int(n / 2) - 2, size=num_extra_edges, replace=False)

	for ii in range(1, n):
		eq = eq + chars[ii - 1] + chars[ii]
		idx = np.where(extra_edges_start_nodes == (ii - 1))[0]
		if len(idx) > 0:
			eq = eq + chars[n + idx[0]]
		idx = np.where(extra_edges_end_nodes == (ii - 1))[0]
		if len(idx) > 0:
			eq = eq + chars[n + idx[0]]

		eq = eq + ','
	eq = eq[:-1]
	eq = eq + f'->{eq[0]}{eq[-1]}'
	input_tensors, output_tensors = str.split(eq, '->')
	# assert len(output_tensors) == 0
	index_per_node = str.split(input_tensors, ',')
	shapes = get_shapes_from_tensors(index_per_node, size_dict)
	return eq, shapes, size_dict


def get_shapes_from_tensors(tensor_list, size_dict):
	shapes = []
	for t in tensor_list:
		shapes.append(tuple([size_dict[x] for x in t]))
	return shapes


def tensors_to_eq(tensors, output_tensor, size_dict):
	eq = [''.join(t) for t in tensors]
	eq = ','.join(eq) + '->' + output_tensor
	shapes = get_shapes_from_tensors(tensors, size_dict)
	return eq, shapes


def get_tensor_size(tensor, size_dict):
	"""
	:param tensor: a tuple of indices ('A','B','C')
	:param size_dict: a dictionary of size {'A':4,....}
	:return:  The tensor size
	"""
	x = math.prod([size_dict[k] for k in tensor])
	x = min(x, 1e30)
	return x
