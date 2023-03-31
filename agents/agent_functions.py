import torch
import torch as th
from torch.nn.functional import pad

DEGREE_CENTRALITY_INDEX = 5


def get_dist(config, score, eval_mode=False, selected_nodes=None, device='cpu'):
	if selected_nodes is None:
		selected_nodes = []
	if config['model']['score_to_prob'] == 'logits':
		dist = th.distributions.categorical.Categorical(logits=score)
	elif config['model']['score_to_prob'] == 'probs':
		# if eval_mode:
		#     score = fill_with_value(score, selected_nodes, value=score.max(), device=device)
		rescaled_score = score - score.min() + config['model']['eps']
		# if eval_mode:
		#     rescaled_score = fill_with_value(rescaled_score, selected_nodes, device=device)
		rescaled_score /= rescaled_score.sum()
		dist = th.distributions.categorical.Categorical(probs=rescaled_score)
	elif config['model']['score_to_prob'] == 'probs_unnormalized':
		if eval_mode:
			score = fill_with_value(score, selected_nodes, device=device)
		abs_score = th.abs(score)
		# norm_term = th.sum(abs_score, dim=1, keepdim=True)
		# rescaled_score = abs_score/norm_term #performed in categorial dist.
		dist = th.distributions.categorical.Categorical(probs=abs_score)
	else:
		raise Exception('Unknown config.model.score_to_prob value')
	return dist


def get_dist_parallel(config, score, edges_list, split_weights):
	max_set_size = torch.max(edges_list)
	if config['model']['score_to_prob'] == 'logits':
		min_score = score.min().abs()
		value = float(-20 * max(1, min_score))
	else:
		value = 0
	if config['model']['score_to_prob'] == 'probs':
		split_weights = [x - torch.min(x) + config['model']['eps'] for x in split_weights]
	padded_weights = [pad(x, pad=(0, max_set_size - len(x)), value=value) for i, x in enumerate(split_weights)]
	padded_weights = th.stack(padded_weights, dim=0)

	if config['model']['score_to_prob'] == 'logits':
		# dist = [th.distributions.categorical.Categorical(logits=x) for x in splitted_weights]
		dist = th.distributions.categorical.Categorical(logits=padded_weights)
	elif config['model']['score_to_prob'] == 'probs':
		dist = th.distributions.categorical.Categorical(probs=padded_weights)
	elif config['model']['score_to_prob'] == 'probs_unnormalized':
		abs_score = th.abs(padded_weights)
		dist = th.distributions.categorical.Categorical(probs=abs_score)
	else:
		raise Exception('Unknown config.model.score_to_prob value')
	return dist


def fill_with_value(score, selected_nodes, value=None, device='cpu'):
	if value:
		fill_value = th.full(selected_nodes.shape, fill_value=value, device=device)
	else:
		fill_value = th.zeros(selected_nodes.shape, device=device)
	score = score.scatter_(1, selected_nodes, fill_value)
	return score
