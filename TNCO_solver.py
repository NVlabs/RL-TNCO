import logging
import os.path
import pickle
import time

import numpy as np

from agents.custom_callback import CuQuantumCallback
from agents.custom_ppo import CuQuantumPPO
from env.tensorNetworkEnv import get_simulator
from gnn.gnn_model import GNN_model
from utils.general_usage_utils import load_torch_rng_state


class TNCOsolver(object):
    def __init__(self, config):
        config['reward_normalization_factor'] = 1.0
        config['eval']['reset_eval_env_after_evaluation'] = False
        config['train_params']['sync_mode'] = 'eval_envs'
        self.callback = CuQuantumCallback(config=config)
        env = get_simulator(config)
        n_steps = int(config['learning']['steps_per_epoch'] / env.num_envs)
        assert config['model']['pretrained_model'], 'TNCO solver uses a pretrained model'
        self.model = CuQuantumPPO(policy=GNN_model, env=env, policy_kwargs={'config': config},
                                  tensorboard_log='./runs', n_steps=n_steps,
                                  batch_size=config['learning']['batch_size'], gamma=config['learning']['gamma'],
                                  ent_coef=config['learning']['entropy_weight'],
                                  vf_coef=config['learning']['value_weight'],
                                  seed=config['train_params']['seed'], device=config['train_params']['device'],
                                  n_epochs=config['train_params']['train_epochs_per_rollout'])
        self.model.set_parameters(config['model']['pretrained_model'], device=config['train_params']['device'])
        stat_file = os.path.join(os.path.dirname(config['model']['pretrained_model']),
                                 'stats' + os.path.basename(config['model']['pretrained_model'])[5:])
        assert os.path.isfile(stat_file), 'stats file must be supplied'
        stats = pickle.load(open(stat_file, 'rb'))
        logging.info('=============Starting Evaluation=======================')
        total_timesteps, self.callback = self.model._setup_learn(total_timesteps=0, callback=self.callback, tb_log_name='PPO')
        self.callback.on_training_start(locals_=None, globals_=None,
                                        external_constant=stats['reward_normalization_factor'])
        self.callback.counter = config['eval']['long_eval_freq']
        if 'numpy_rng_state' in stats:
            np.random.set_state(stats['numpy_rng_state'])
        if 'torch_rng_state' in stats:
            load_torch_rng_state(stats['torch_rng_state'][0], gpu_rng_state=stats['torch_rng_state'][1],
                                 device=self.model.device)

    def find_path(self):
        start_time = time.time()
        self.callback.evaluate_model()
        finish_time = time.time()
        best_result = self.callback.update_best_result()
        print(f'Path finding took {finish_time - start_time}')
        return best_result
