import logging
import numpy as np


def maybe_update_best_result(best_result, candidate_result):
	if candidate_result is not None:
		if isinstance(best_result, dict):
			return maybe_update_best_result_per_eq(best_result, candidate_result)
		elif isinstance(best_result, list):
			assert len(best_result) == len(candidate_result)
			best_result = [maybe_update_best_result_per_eq(best_result[i], candidate_result[i])
			               for i in range(len(candidate_result))]
		elif best_result is None:
			best_result = candidate_result
	return best_result


def maybe_update_best_result_per_eq(best_result, candidate_result):
	if candidate_result['total_reward'] > best_result['total_reward'] and not candidate_result['baseline_policy']:
		best_result = candidate_result
		if best_result is not None and 'history_by_indices' in candidate_result:
			assert len(candidate_result['history_by_indices']) == len(best_result['history_by_indices']), \
				'Replacing by shorter path!'
	return best_result


def get_best_result(env, same_eq_subsets=None):
	# candidate_result_array = env.get_best_result()
	candidate_result_array = env.get_attr('best_result')
	if same_eq_subsets:
		best_result_list = []
		for same_eq_envs in same_eq_subsets:
			best_result = None
			for ind in same_eq_envs:
				best_result = maybe_update_best_result(best_result, candidate_result_array[ind])
			best_result_list.append(best_result)
		return best_result_list
	else:
		best_result = None
		for ind in candidate_result_array:
			best_result = maybe_update_best_result(best_result, ind)
		return best_result
