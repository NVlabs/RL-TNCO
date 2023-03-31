from collections import namedtuple

import torch
# from gnn.sampler import Sampler
import wandb
from stable_baselines3.common.policies import BasePolicy
from torch_geometric.data import Data, Batch
from torch_geometric.nn import DataParallel

from agents.agent_functions import get_dist, get_dist_parallel
from gnn.gnn import GNN

BatchTuple = namedtuple('BatchTuple', 'X edge_index edge_attr num_nodes u custom_batch feasible_actions')


def _get_sample(i, state):
	n_nodes = int(state['n_nodes'][i])
	n_edges = int(state['n_edges'][i])
	X = state['X'][i, :n_nodes]
	edge_features = state['edge_features'][i, :n_edges]
	edge_list = state['edge_index'][i, :, :n_edges]
	edge_list_reversed = torch.stack([edge_list[1, :], edge_list[0, :]], dim=0)
	edge_list = torch.cat([edge_list, edge_list_reversed], dim=1)
	feasible_actions = state['feasible_actions'][i, :n_edges]
	edge_attr = edge_features.repeat(2, 1)
	sample = Data(X=X,
	              edge_index=edge_list,
	              edge_attr=edge_attr,
	              u=state['global_features'][i],
	              custom_batch=torch.zeros(n_nodes, dtype=int),
	              n_edges=n_edges,
	              num_nodes=n_nodes,
	              feasible_actions=feasible_actions)
	return sample


class GNN_model(BasePolicy):
	def __init__(self, observation_space, action_space, lr_schedule=None, use_sde=False, config=None, **kwargs):
		self.blocked_action_index = -1
		self.n_nodes = kwargs.get('n_nodes', config['network']['n_nodes'])
		device = 'cuda' if config['train_params']['device'] != 'cpu' else 'cpu'
		torch.nn.Module.__init__(self)
		torch.manual_seed(config['train_params']['seed'])
		torch.cuda.manual_seed_all(config['train_params']['seed'])
		self.config = config
		self.multi_gpu = config['train_params']['use_multi_GPU']
		self.device_count = torch.cuda.device_count()
		self.block_decreasing_actions = config['train_params']['block_decreasing_action']
		if self.multi_gpu:
			self.actor = DataParallel(
				GNN(config, observation_space=observation_space, value_net=False, device=device)).to(device)
			self.v = DataParallel(GNN(config, observation_space=observation_space, value_net=True, device=device)).to(
				device)
		else:
			self.actor = GNN(config, observation_space=observation_space, value_net=False, device=device).to(device)
			self.v = GNN(config, observation_space=observation_space, value_net=True, device=device).to(device)

		wandb.watch(self.actor)
		self.actor.train()
		wandb.watch(self.v)
		self.v.train()
		self._value_out = None
		self.iteration = 0
		self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning']['pi_lr'])
		self.v_optimizer = torch.optim.Adam(self.parameters(), lr=config['learning']['pi_lr'])
		self.get_value = True
		self.counter = 0
		self.greedy_index = 0

	def forward(self, state, deterministic=False):
		batch, batch_mode = self._get_batch(state)
		logits = self.actor(batch)
		n_edges = logits.shape[0]
		if self.block_decreasing_actions:
			# filter the blocked logits
			feasible_actions = batch.feasible_actions[:n_edges].type(torch.bool).to('cpu')
			logits = logits[feasible_actions]

		# use greedy score to define a distributation based on a mixture between the greedy score and the logits
		greedy_weight = self.config['representation']['greedy_weight']
		if greedy_weight > 0:
			greedy_score = batch.edge_attr[:n_edges, self.greedy_index:self.greedy_index + 1]
			if self.block_decreasing_actions:
				greedy_score = greedy_score[feasible_actions]
			logits = logits + greedy_weight * greedy_score

		if batch_mode:
			if self.block_decreasing_actions:
				# filter the blocked edges
				elements_per_sample = state['feasible_actions'].sum(dim=1).type(torch.int)
				edge_index_list = [torch.arange(x, dtype=torch.int) for x in state['n_edges'].squeeze()]
				full_indices = torch.cat(edge_index_list)
				filtered_indices = torch.split(full_indices[feasible_actions], elements_per_sample.tolist())
			else:
				elements_per_sample = state['n_edges'].flatten()
				filtered_indices = None
			actions, log_probs, _ = self.parallel_sample(scores=logits, elements_per_sample=elements_per_sample,
			                                             deterministic=deterministic, filtered_indices=filtered_indices)
		else:
			actions, log_probs, _ = self.sample(inputs=logits, deterministic=deterministic)
			if self.block_decreasing_actions:
				indices = feasible_actions.nonzero()
				actions = indices[actions]
		values = self.v(batch)
		return actions, values, log_probs

	def _get_batch(self, state):
		sample_size = state['X'].shape[0]
		batch_mode = sample_size > 1
		if self.multi_gpu:
			if batch_mode:
				batch = self.build_batch_multi_gpu(state, sample_size)
			else:
				batch = self._build_batch(state, return_list=True)
		else:
			if batch_mode:
				batch = self._build_batch_parallel(state)
			else:
				batch = self._build_batch(state)
		return batch, batch_mode

	def predict_values(self, state):
		batch, batch_mode = self._get_batch(state)
		values = self.v(batch)
		return values

	def build_batch_multi_gpu(self, state, sample_size):
		batch = []
		range_limit = sample_size // self.device_count
		ranges = torch.cat([range_limit * torch.arange(self.device_count), torch.tensor([sample_size])])
		for i in range((len(ranges) - 1)):
			partial_state = {'X': state['X'][ranges[i]:ranges[i + 1]],
			                 'edge_features': state['edge_features'][ranges[i]:ranges[i + 1]],
			                 'edge_index': state['edge_index'][ranges[i]:ranges[i + 1]],
			                 'global_features': state['global_features'][ranges[i]:ranges[i + 1]],
			                 'n_edges': state['n_edges'][ranges[i]:ranges[i + 1]],
			                 'n_nodes': state['n_nodes'][ranges[i]:ranges[i + 1]],
			                 }
			data = self._build_batch_parallel(partial_state)
			batch.append(Data(X=data.X, edge_attr=data.edge_attr, edge_index=data.edge_index, num_nodes=data.num_nodes,
			                  u=data.u, custom_batch=data.custom_batch))
		return batch

	def _build_batch(self, state, return_list=False):
		state['edge_index'] = state['edge_index'].type(torch.long)
		batch = [_get_sample(0, state)]
		if not return_list:
			batch = Batch.from_data_list(batch).to(self.device)
		return batch

	def _build_batch_parallel(self, state):
		"""
		Constructs a graph composed of n disconnected subgraphs. Performs the graph construction much faster than the
		corresponding pytorch Geometric code.
		"""

		n_envs, max_nodes, _ = state['X'].shape
		max_edges = state['edge_index'].shape[2]
		accumalted_initial_nodes_index = torch.cumsum(state['n_nodes'][:-1], dim=0)
		accumalted_initial_nodes_index = torch.cat(
			[torch.tensor([[0]], device=self.device), accumalted_initial_nodes_index])
		idx = torch.arange(max_edges, device=self.device).unsqueeze(0).expand(n_envs, -1)
		idx_edges = idx < state['n_edges']
		edge_features = state['edge_features'][idx_edges]
		edge_features = edge_features.repeat(2, 1)
		if self.block_decreasing_actions:
			feasible_actions = state['feasible_actions'][idx_edges]
		else:
			feasible_actions = None

		edge_index = state['edge_index'] + accumalted_initial_nodes_index.unsqueeze(dim=2)
		edge_index = edge_index.transpose(1, 2)[idx_edges].transpose(0, 1)

		edge_list_reversed = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
		edge_list = torch.cat([edge_index, edge_list_reversed], dim=1)

		idx = torch.arange(max_nodes).unsqueeze(0).expand(n_envs, -1)
		idx_nodes = idx < state['n_nodes'].to(idx.device)
		node_features = state['X'][idx_nodes]

		batch_array = torch.arange(n_envs).unsqueeze(1).expand(-1, max_nodes)
		batch_tensor = batch_array[idx_nodes].to(device=self.device)

		batch = BatchTuple(X=node_features, edge_index=edge_list.type(torch.long), edge_attr=edge_features,
		                   num_nodes=batch_tensor.shape[0], u=state['global_features'].squeeze(dim=1),
		                   custom_batch=batch_tensor,
		                   feasible_actions=feasible_actions)
		return batch

	def evaluate_actions(self, state, actions, value_only=False):
		sample_size = state['X'].shape[0]
		if self.multi_gpu:
			batch = self.build_batch_multi_gpu(state, sample_size)
		else:
			batch = self._build_batch_parallel(state)
		if value_only:
			log_prob, entropy = None, None
		else:
			logits = self.actor(batch)
			action, log_prob, entropy = \
				self.parallel_sample(logits, elements_per_sample=state['n_edges'].flatten(), action=actions)
		values = self.v(batch)
		return values, log_prob, entropy

	def predict(self, obs, state=None, mask=None, deterministic=True):
		obs['X'] = torch.tensor(obs['X'], device=self.device)
		obs['edge_features'] = torch.tensor(obs['edge_features'], device=self.device)
		obs['global_features'] = torch.tensor(obs['global_features'], device=self.device)
		obs['edge_index'] = torch.tensor(obs['edge_index'], device=self.device)
		with torch.no_grad():
			actions, values, log_probs = self.forward(obs, deterministic=deterministic)
		model_info = {'value': values,
		              'log_probs': log_probs}
		return actions, model_info

	def _predict(self, obs, state=None, mask=None, deterministic=True):
		raise NotImplementedError

	def parallel_sample(self, scores, elements_per_sample, action=None, deterministic=False, filtered_indices=None):
		# split the large graph into connected graphs (each graph correspond to a sample in the batch)
		edges_list = elements_per_sample.flatten().to(int)
		split_scores = torch.split(scores.flatten(), edges_list.tolist())

		if deterministic:
			action = [int(torch.argmax(x, dim=0).cpu()) for x in split_scores]
			entropy = 0
			log_prob = 0
		else:
			dist = get_dist_parallel(self.config, scores.flatten(), edges_list, split_scores)
			if action is None:
				action = dist.sample([1]).squeeze()
			log_prob = dist.log_prob(action)
			entropy = dist.entropy().squeeze()
		if filtered_indices is not None:
			action = torch.tensor([filtered_indices[i][action[i]] for i in range(len(action))])
		action = action.to('cpu')
		return action, log_prob, entropy

	def sample(self, inputs, action=None, deterministic=False):
		device = self.device
		if deterministic:
			action = torch.argmax(inputs, dim=0)
			entropy = 0
			log_prob = 0
		else:
			dist = get_dist(self.config, inputs.flatten(), eval_mode=False, device=device)
			if action is None:
				action = dist.sample([1])
			log_prob = dist.log_prob(action)
			entropy = dist.entropy()
		action = action.to('cpu')
		return action, log_prob, entropy
