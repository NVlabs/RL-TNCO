from typing import Optional

import torch
from torch import Tensor
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.norm import pair_norm
from torch_scatter import scatter_mean, scatter_max, scatter_sum


class EdgeModel(torch.nn.Module):
	def __init__(self, num_input_node_features, num_input_edge_features, num_input_global_features,
	             num_intermediate_features, num_output_edge_features, use_bias, use_batch_norm=False):
		super(EdgeModel, self).__init__()
		# takes as input an edge feature and two adjacent node features
		if not use_batch_norm:
			self.edge_mlp = Seq(Lin(2 * num_input_node_features + num_input_edge_features + num_input_global_features,
			                        num_intermediate_features, bias=use_bias), ReLU(), Lin(num_intermediate_features,
			                                                                               num_output_edge_features,
			                                                                               bias=use_bias))
		else:
			self.edge_mlp = Seq(Lin(2 * num_input_node_features + num_input_edge_features + num_input_global_features,
			                        num_intermediate_features, bias=use_bias), BatchNorm1d(num_intermediate_features),
			                    ReLU(), Lin(num_intermediate_features,
			                                num_output_edge_features,
			                                bias=use_bias))

	def forward(self, src: Tensor, dest: Tensor,
	            edge_attr: Optional[Tensor], u: Optional[Tensor],
	            batch: Optional[Tensor]) -> Tensor:
		assert edge_attr is not None
		assert u is not None
		assert batch is not None
		out = torch.cat([src, dest, edge_attr, u[batch]], 1)
		return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
	def __init__(self, num_input_node_features, num_output_edge_features, num_input_global_features,
	             num_intermediate_features, num_output_node_features, agg_type='mean', use_bias=True,
	             use_batch_norm=False):
		super(NodeModel, self).__init__()
		self.agg_type = agg_type
		if not use_batch_norm:
			self.node_mlp_1 = Seq(Lin(num_output_edge_features + num_input_node_features,
			                          num_intermediate_features, bias=use_bias), ReLU(), Lin(num_intermediate_features,
			                                                                                 num_intermediate_features,
			                                                                                 bias=use_bias))
			self.node_mlp_2 = Seq(
				Lin(num_intermediate_features + num_input_node_features + num_input_global_features,
				    num_intermediate_features,
				    bias=use_bias),
				ReLU(), Lin(num_intermediate_features, num_output_node_features, bias=use_bias))
		else:
			self.node_mlp_1 = Seq(Lin(num_output_edge_features + num_input_node_features,
			                          num_intermediate_features, bias=use_bias), BatchNorm1d(num_intermediate_features),
			                      ReLU(), Lin(num_intermediate_features,
			                                  num_intermediate_features,
			                                  bias=use_bias))
			self.node_mlp_2 = Seq(
				Lin(num_intermediate_features + num_input_node_features + num_input_global_features,
				    num_intermediate_features,
				    bias=use_bias), BatchNorm1d(num_intermediate_features),
				ReLU(), Lin(num_intermediate_features, num_output_node_features, bias=use_bias))

	def forward(self, x: Tensor, edge_index: Tensor,
	            edge_attr: Optional[Tensor], u: Optional[Tensor],
	            batch: Optional[Tensor]) -> Tensor:
		assert edge_attr is not None
		assert u is not None
		assert batch is not None
		row = edge_index[0]
		col = edge_index[1]
		out = torch.cat([x[row], edge_attr], dim=1)
		out = self.node_mlp_1(out)
		if self.agg_type == 'mean':
			out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
		elif self.agg_type == 'max':
			out = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
		elif self.agg_type == 'sum':
			out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
		else:
			raise NotImplemented
		out = torch.cat([x, out, u[batch]], dim=1)
		return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
	def __init__(self, num_input_global_features, num_intermediate_features, num_output_node_features,
	             num_output_global_features, agg_type='mean', use_bias=True, use_batch_norm=False):
		super(GlobalModel, self).__init__()
		self.agg_type = agg_type
		if not use_batch_norm:
			self.global_mlp = Seq(
				Lin(num_input_global_features + num_output_node_features, num_intermediate_features, bias=use_bias),
				ReLU(), Lin(num_intermediate_features, num_output_global_features, bias=use_bias))
		else:
			self.global_mlp = Seq(
				Lin(num_input_global_features + num_output_node_features, num_intermediate_features, bias=use_bias),
				BatchNorm1d(num_intermediate_features), ReLU(),
				Lin(num_intermediate_features, num_output_global_features, bias=use_bias))

	def forward(self, x: Tensor, edge_index: Tensor,
	            edge_attr: Optional[Tensor], u: Optional[Tensor],
	            batch: Optional[Tensor]) -> Tensor:
		assert u is not None
		assert batch is not None
		if self.agg_type == 'mean':
			out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
		elif self.agg_type == 'max':
			out = torch.cat([u, scatter_max(x, batch, dim=0)[0]], dim=1)
		elif self.agg_type == 'sum':
			out = torch.cat([u, scatter_sum(x, batch, dim=0)], dim=1)
		else:
			raise NotImplemented
		return self.global_mlp(out)


def create_meta_layer(num_input_edge_features, num_input_global_features, num_input_node_features,
                      num_intermediate_features, num_output_edge_features, num_output_global_features,
                      num_output_node_features, agg_type='mean', use_bias=True, use_batch_norm=False):
	op = MetaLayer(
		EdgeModel(num_input_node_features, num_input_edge_features, num_input_global_features,
		          num_intermediate_features, num_output_edge_features,
		          use_bias=use_bias, use_batch_norm=use_batch_norm),
		NodeModel(num_input_node_features, num_output_edge_features, num_input_global_features,
		          num_intermediate_features, num_output_node_features, agg_type, use_bias=use_bias,
		          use_batch_norm=use_batch_norm),
		GlobalModel(num_input_global_features, num_intermediate_features, num_output_node_features,
		            num_output_global_features, agg_type, use_bias=use_bias, use_batch_norm=use_batch_norm))
	return op


class GNN(torch.nn.Module):
	def __init__(self, config, observation_space, value_net=False, device='cpu'):
		super(GNN, self).__init__()
		num_input_edge_features = observation_space['edge_features'].shape[1]
		num_input_global_features = observation_space['global_features'].shape[1]
		num_input_node_features = observation_space['X'].shape[1]
		num_gnn_layers = config['model']['num_gnn_layers']
		num_intermediate_features = config['model']['num_intermediate_features']
		use_batch_norm = config['model']['use_batch_norm']
		self.use_pair_norm = config['model']['use_pair_norm']

		if self.use_pair_norm:
			self.pairnorm = pair_norm.PairNorm()
		self.use_bias = config['model']['use_bias']
		self.value_net = value_net

		self.layers = torch.nn.ModuleList()
		self.layers.append(create_meta_layer(
			num_input_edge_features=num_input_edge_features,
			num_input_global_features=num_input_global_features,
			num_input_node_features=num_input_node_features,
			num_intermediate_features=num_intermediate_features,
			num_output_edge_features=num_intermediate_features,
			num_output_global_features=num_intermediate_features,
			num_output_node_features=num_intermediate_features, agg_type=config['model']['agg_type'],
			use_bias=self.use_bias, use_batch_norm=use_batch_norm),
		)
		for i in range(num_gnn_layers - 2):
			self.layers.append(create_meta_layer(
				num_input_edge_features=num_intermediate_features,
				num_input_global_features=num_intermediate_features,
				num_input_node_features=num_intermediate_features,
				num_intermediate_features=num_intermediate_features,
				num_output_edge_features=num_intermediate_features,
				num_output_global_features=num_intermediate_features,
				num_output_node_features=num_intermediate_features, agg_type=config['model']['agg_type'],
				use_bias=self.use_bias, use_batch_norm=use_batch_norm))

		if value_net:
			num_output_global_features = 1
			num_output_edge_features = num_intermediate_features
			num_output_node_features = num_intermediate_features
		else:
			num_output_global_features = 1
			num_output_edge_features = 1
			num_output_node_features = 1

		self.layers.append(create_meta_layer(
			num_input_edge_features=num_intermediate_features,
			num_input_global_features=num_intermediate_features,
			num_input_node_features=num_intermediate_features,
			num_intermediate_features=num_intermediate_features,
			num_output_edge_features=num_output_edge_features,
			num_output_global_features=num_output_global_features,
			num_output_node_features=num_output_node_features, agg_type=config['model']['agg_type'],
			use_bias=self.use_bias, use_batch_norm=use_batch_norm))

	def forward(self, batch):
		X, edge_index, edge_attr, u, batch = batch.X, batch.edge_index, batch.edge_attr, batch.u, batch.custom_batch
		for layer in self.layers:
			X, edge_attr, u = layer(X, edge_index, edge_attr, u, batch)
			if self.use_pair_norm:
				X = self.pairnorm(X, batch)
		if self.value_net:
			return u.squeeze()
		else:
			n_edges = int(edge_attr.shape[0] / 2)
			logits = (edge_attr[:n_edges] + edge_attr[n_edges:]) / 2
			return logits
