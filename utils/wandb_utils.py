import logging
import os
import subprocess

import numpy as np
import wandb

from config import get_config, create_logger
from datetime import datetime


def assignment_allowed(value, target_value):
    both_numerics = isinstance(target_value, (int, float, complex)) and isinstance(value, (int, float, complex))
    same_type = type(value) is type(target_value) or value is None or target_value is None
    assigment_allowed = same_type or both_numerics
    return assigment_allowed


def assign_value(location_string, target_value, value, force_print=False):
    if target_value == value:
        if force_print:
            logging.info('Updating key {} to: {} (default value: {}))'.format(location_string, target_value, value))
        return value
    assigment_allowed = assignment_allowed(value, target_value)
    if assigment_allowed:
        logging.info('Updating key {} to: {} (default value: {}))'.format(location_string, target_value, value))
        res = target_value
    else:
        logging.warning('Incorrect key type {}. Ignored assignment: {}, type: {} (default value: {}, '
                        'type: {}))'.format(location_string, target_value, type(target_value), value, type(value)))
        res = value
    return res


def update_keys(source, target):
    for key, subdict in source.items():
        if not (isinstance(subdict, dict)):
            source[key] = assign_value(key, target[key], source[key])
            continue
        for subkey, value in subdict.items():
            target_value = target[key][subkey]
            subkey_str = '[{}][{}]'.format(key, subkey)
            source[key][subkey] = assign_value(subkey_str, target_value, value)
    new_keys = target.keys() - source.keys()
    for key in new_keys:
        assigned = False
        splitted_key = key.split('.')
        plain_key = splitted_key[0] == key
        for root_key, subdict in source.items():
            if isinstance(subdict, dict) and ((key in subdict) or (not plain_key and splitted_key[1] in subdict)):
                target_key = key if key in subdict else splitted_key[1]
                target_value = target[key]
                source_value = subdict[target_key]
                subkey_str = '[{}][{}]'.format(root_key, target_key)
                subdict[target_key] = assign_value(subkey_str, target_value, source_value, force_print=True)
                assigned = True
                break
        if not assigned:
            logging.warning('Unable to assign {}. Ignored assignment: {})'.format(key, target[key]))
    return source


def presetup_experiment():
    config = get_config()
    create_logger(config)
    try:
        wandb.init(project=config['wandb']['project_name'], sync_tensorboard=True, config=config)
        config = update_keys(config, wandb.config)
    except:
        now = datetime.now()  # current date and time
        time_str = now.strftime("%m_%d_%H_%M_%S")
        wandb.init(mode='disabled', dir=os.path.join('runs', time_str))
    cpu_info = subprocess.run('cat /proc/cpuinfo | grep "model name" | head -n 1', shell=True, executable="/bin/bash",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    logging.info("CPU Info: " + cpu_info.stdout)
    return config


def get_best_result(callback, model):
    callback.update_best_result()
    best_short_eval_result = abs(np.mean([x['total_reward'] for x in model.best_results['eval']]))
    best_long_eval_result = abs(np.mean([x['total_reward'] for x in model.best_results['long_eval']]))
    best_result = min(best_short_eval_result, best_long_eval_result)
    return best_result
