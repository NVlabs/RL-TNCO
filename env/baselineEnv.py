import logging

import numpy as np
from gym.spaces import Box

from env.baseEnv import TensorNetworkEnv, tensors_to_eq


class BaselineEnv(TensorNetworkEnv):
	def __init__(self, config, solver, **kwargs):
		super().__init__(config, **kwargs)
		self.output_tensor = None
		self.size_dict = None
		self.shapes = None
		self.eq = None
		assert self.baseline_solutions is not None, 'Incorrect configuration for baseline env'
		self.solver = solver
		self.solver_path = None
		self.step_counter = None
		self.observation_space = Box(high=np.inf, low=-np.inf, shape=(0,))

	def reset(self):
		self.new_episode = True
		self.step_counter = 0
		if self.video_writer is not None:
			self.finish_video()
		if self.data:
			logging.info('using obsolete code!')
			self.tensors, size_dict = self.data['tensors'], self.data['size_dict']
			eq, shapes = self.data['eq'], self.data['shapes']
			self.output_tensor = ''
			self.data = None
		else:
			if self.use_fixed_eqs_set:
				eq, shapes, size_dict = self.eqs[self.counter]
			else:
				eq, shapes, size_dict = self.get_eq_shapes_and_sizes()
			index_per_node, self.output_tensor = self.split_eq_to_tensors(eq)
			self.tensors = [tuple(x) for x in index_per_node]
		self.eq = eq
		self.shapes = shapes
		self.size_dict = size_dict
		self.state = True
		self.episode_history = []
		cost, _, info, path = self.baseline_solutions[self.solver][self.counter]
		self.solver_path = path
		return self.state

	def step(self, action=None, plot=False):
		u, v = sorted(self.solver_path[self.step_counter])
		scalar_multiplication = len(self.tensors) <= v
		if scalar_multiplication:
			reward = 0
			rewards_info = {'flops_only': 0}
		else:
			new_tensor = tuple(set(self.tensors[u]).symmetric_difference(set(self.tensors[v])))
			reward, rewards_info = self.get_reward(u, v, new_tensor)
		self.episode_history.append((u, v))
		if not scalar_multiplication:
			self.tensors.pop(v)
			self.tensors.pop(u)
			self.tensors.append(new_tensor)
			self.eq, self.shapes = tensors_to_eq(self.tensors, self.output_tensor, self.size_dict)
		self.step_counter += 1
		done = self.step_counter == len(self.solver_path)
		if done and self.eqs:
			self.counter = (self.counter + 1) % len(self.eqs)
		self.new_episode = False
		return self.state, reward, done, rewards_info


class FollowThePathEnv(TensorNetworkEnv):
	def __init__(self, config, **kwargs):
		super().__init__(config, **kwargs)
		self.solver_path = None
		self.step_counter = None
		self.observation_space = Box(high=np.inf, low=-np.inf, shape=(0,))
		self.size_dict = None
		self.shapes = None
		self.eq = None

	def reset(self, tensors, eq, shapes, size_dict, path):
		self.new_episode = True
		self.step_counter = 0
		self.tensors = tensors
		self.eq = eq
		self.shapes = shapes
		self.size_dict = size_dict
		self.state = True
		self.episode_history = []
		self.solver_path = path
		return self.state

	def step(self, action=None, plot=False):
		u, v = sorted(self.solver_path[self.step_counter])
		scalar_multiplication = len(self.tensors) <= v
		if scalar_multiplication:
			reward = 0
			rewards_info = {'flops_only': 0}
		else:
			new_tensor = tuple(set(self.tensors[u]).symmetric_difference(set(self.tensors[v])))
			reward, rewards_info = self.get_reward(u, v, new_tensor)
		self.episode_history.append((u, v))
		if not scalar_multiplication:
			self.tensors.pop(v)
			self.tensors.pop(u)
			self.tensors.append(new_tensor)
		self.step_counter += 1
		done = self.step_counter == len(self.solver_path)
		self.new_episode = False
		return self.state, reward, done, rewards_info
