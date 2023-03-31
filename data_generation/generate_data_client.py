import argparse
import os
import pickle
import timeit

import cotengra as ctg
import opt_einsum as oe

import generate_data

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--eq_number', type=int, help='an integer for the accumulator', default=1)
parser.add_argument('--baseline', help='an integer for the accumulator', default='oe_greedy')
parser.add_argument('--output_dir', help='output dir', default='.')
parser.add_argument('--equation_fname', help='equation file', default='.')


def solve_eq(eq, shapes, baseline):
	start = timeit.default_timer()
	if baseline == 'ctg_kahypar':
		opt = ctg.HyperOptimizer(methods=['kahypar'], max_repeats=1000)
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
	elif baseline == 'ctg_greedy':
		opt = ctg.HyperOptimizer(methods=['greedy'], max_repeats=1000)
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
	elif baseline == 'oe_greedy':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='greedy')
	elif baseline == 'oe_dp':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='dp')
	elif baseline == 'oe_optimal':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='optimal')
	elif baseline == 'oe_branch-1':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='branch-1')
	elif baseline == 'oe_branch-2':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='branch-2')
	elif baseline == 'oe_branch-all':
		path, info = oe.contract_path(eq, *shapes, shapes=True, optimize='branch-all')
	else:
		print('undefined baseline')
	stop = timeit.default_timer()
	total_time = stop - start
	contraction_cost = info.opt_cost
	print('solving with {} yielded contraction_cost={} in {} sec.'
	      .format(baseline, contraction_cost, total_time))
	return contraction_cost, total_time, info, path


def main():
	args = parser.parse_args()

	print('loading eq {}'.format(args.eq_number))
	equation = pickle.load(open(os.path.join(args.output_dir, args.equation_fname), "rb"))
	contraction_cost, solver_time, info, path = solve_eq(equation['eq'], equation['shapes'],
	                                                     baseline=args.baseline)
	sol = {'contraction_cost': contraction_cost, "solver_time": solver_time, "info": info, 'path': path}
	sys_info = generate_data.getSystemInfo()

	pickle.dump({'args': args, 'equation': equation, 'baseline': args.baseline, 'sol': sol, 'sys_info': sys_info},
	            open(os.path.join(args.output_dir, "eq_{}_{}_result.p".format(args.eq_number, args.baseline)), "wb"))


if __name__ == '__main__':
	main()
