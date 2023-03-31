import json
import logging
import os
import pickle
import platform
from collections import defaultdict

import psutil
import re
import socket
import timeit
import uuid

import cotengra as ctg
import subprocess

import opt_einsum as oe
from tqdm import tqdm
import wandb
from config import get_config, create_logger
from generate_data import getSystemInfo, solve_eq
from utils.main_utils import read_data_file
from utils.wandb_utils import update_keys


def evaluate_eq_dataset(data, time_limit, baseline, filename='test.p'):
	solution_dict = defaultdict(list)
	for eq, shape, _ in tqdm(data):
		contraction_cost, solver_time, info, path = solve_eq(eq, shape, max_time=time_limit, baseline=baseline)
		solution_dict[baseline].append((contraction_cost, solver_time, info, path))
	sys_info = getSystemInfo()
	pickle.dump((solution_dict, sys_info), open(filename, "wb"))
	wandb.save(filename)


json.loads(getSystemInfo())

if __name__ == '__main__':
	config, eval_config = get_config()
	create_logger(config)
	ngc_run = os.path.isdir('/ws')
	if ngc_run:
		ngc_dir = '/result/wandb/'  # args.ngc_path
		os.makedirs(ngc_dir, exist_ok=True)
		logging.info('NGC run detected. Setting path to workspace: {}'.format(ngc_dir))
	if ngc_run:
		wandb.init(project="il-cuquantum", sync_tensorboard=True, config=config, dir=ngc_dir)
	else:
		wandb.init(project="il-cuquantum", sync_tensorboard=True, config=config)
	cpu_info = subprocess.run('cat /proc/cpuinfo | grep "model name" | head -n 1', shell=True, executable="/bin/bash",
	                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	logging.info('Detected NGC run: {}'.format(ngc_run))
	if 'NGC_JOB_ID' in os.environ:
		logging.info('Job ID: {}'.format(os.environ['NGC_JOB_ID']))
		wandb.config.update({'Job ID': os.environ['NGC_JOB_ID']})
	logging.info("CPU Info: " + cpu_info.stdout)
	config = update_keys(config, wandb.config)
	config_kwarg = {'config': config}
	_, eval_file = read_data_file(config)
	with open(eval_file, 'rb') as f:
		data, _, _ = pickle.load(f)
	time_limit = config['baselines']['time_limit']
	baseline = config['baselines']['baseline']
	if config['baselines']['split'] is not None:
		split_size = len(data) / 5
		i = config['baselines']['split']
		data = data[int(i * split_size):int((i + 1) * split_size)]
		filename = f"baseline_{baseline}_time_limit_{time_limit}_split_{i}.p"
	else:
		filename = f"baseline_{baseline}_time_limit_{time_limit}.p"
	evaluate_eq_dataset(data, baseline=baseline, time_limit=time_limit, filename=filename)
