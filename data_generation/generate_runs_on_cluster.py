import argparse
import os
import pickle
import shutil

import opt_einsum as oe

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_eqs', type=int, help='num_eqs', default=1)
parser.add_argument('--num_nodes', type=int, help='num_nodes', default=10)
parser.add_argument('--mean_connectivity', type=int, help='mean_connectivity', default=3)
parser.add_argument('--q_name', help='queue name', default='o_cpu_2G_15M')

args = parser.parse_args()


def main():
	num_eqs = args.num_eqs
	num_nodes = args.num_nodes
	mean_connectivity = args.mean_connectivity
	# baseline_list = ['ctg_greedy','ctg_kahypar', 'oe_greedy', 'oe_branch-1']
	baseline_list = ['ctg_kahypar']
	python_commands = []
	# create output dir
	output_dir = os.path.join('../data', "TN_d_2_dataset_num_eqs_{}_num_node_{}_mean_conn_{}".
	                          format(num_eqs, num_nodes, mean_connectivity))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	for ii in range(num_eqs):
		print('Generating equation and creating script for eq {}'.format(ii))
		eq, shapes, size_dict = oe.helpers.rand_equation(n=num_nodes, reg=mean_connectivity, d_min=2, d_max=2,
		                                                 return_size_dict=True)
		equation = {'eq_number': ii, 'eq': eq, 'shapes': shapes, 'size_dict': size_dict}
		equation_fname = "eq_{}.p".format(ii)
		pickle.dump(equation, open(os.path.join(output_dir, equation_fname), "wb"))

		for baseline in baseline_list:
			python_command = 'python3 generate_data_client.py --eq_number {}' \
			                 ' --baseline {} --output_dir {} --equation_fname {}'. \
				format(ii, baseline, output_dir, equation_fname)
			python_commands.append(python_command)
	for python_command in python_commands:
		template = 'qsub -q {} --projectMode direct -P research_par_misc "{}"'
		python_command = template.format(args.q_name, python_command)
		print(python_command)
		os.system(python_command)


if __name__ == '__main__':
	main()
