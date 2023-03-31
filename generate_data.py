import json
import logging
import os
import pickle
import platform
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
from utils.wandb_utils import update_keys


def generate_eq_dataset(num_eqs=100, num_nodes=25, mean_connectivity=3, d_min=2, d_max=6):
    eq_list = list()
    baseline_list = ['ctg_greedy', 'ctg_kahypar', 'oe_greedy']

    solution_dict = dict()
    for baseline in baseline_list:
        solution_dict[baseline] = list()

    for ii in tqdm(range(num_eqs)):
        print('processing eq {}'.format(ii))
        eq, shapes, size_dict = oe.helpers.rand_equation(n=num_nodes, reg=mean_connectivity, d_min=d_min, d_max=d_max,
                                                         return_size_dict=True)
        eq_list.append((eq, shapes, size_dict))
        for baseline in baseline_list:
            contraction_cost, solver_time, info, path = solve_eq(eq, shapes, baseline=baseline)
            solution_dict[baseline].append((contraction_cost, solver_time, info, path))
    sys_info = getSystemInfo()
    filename = "for_paper_TN_d_{}_{}_dataset_num_eqs_{}_num_node_{}_mean_conn_{}.p".\
        format(d_min, d_max, num_eqs, num_nodes, mean_connectivity)
    pickle.dump((eq_list, solution_dict, sys_info), open(filename, "wb"))
    wandb.save(filename)


def getSystemInfo():
    try:
        info = {'platform': platform.system(),
                'platform-release': platform.release(),
                'platform-version': platform.version(),
                'architecture': platform.machine(),
                'hostname': socket.gethostname(),
                'ip-address': socket.gethostbyname(socket.gethostname()),
                'mac-address': ':'.join(re.findall('', '%012x' % uuid.getnode())),
                'processor': platform.processor(),
                'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"}
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)


json.loads(getSystemInfo())


def solve_eq(eq, shapes, baseline, max_time=300):
    start = timeit.default_timer()
    if baseline == 'ctg_kahypar':
        opt = ctg.HyperOptimizer(methods=['kahypar'], max_repeats=1000000, max_time=max_time)
        path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
    elif baseline == 'ctg_greedy':
        opt = ctg.HyperOptimizer(methods=['greedy'], max_repeats=1000000, max_time=max_time)
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
        raise  ValueError('undefined baseline')
    stop = timeit.default_timer()
    total_time = stop - start
    contraction_cost = info.opt_cost
    print('solving with {} yielded contraction_cost={} in {} sec.'
          .format(baseline, contraction_cost, total_time))
    return contraction_cost, total_time, info, path


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
    num_eqs = config['baselines']['num_eqs']
    num_nodes = config['baselines']['num_nodes']
    generate_eq_dataset(num_eqs=num_eqs, num_nodes=num_nodes, mean_connectivity=3, d_min=2, d_max=6)
