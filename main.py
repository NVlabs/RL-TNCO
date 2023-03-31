import logging
import numpy as np
import os
import pickle
import torch.random

from agents.custom_callback import CuQuantumCallback
from agents.custom_ppo import CuQuantumPPO
from env.tensorNetworkEnv import get_simulator
from gnn.gnn_model import GNN_model
from utils.general_usage_utils import load_torch_rng_state
from utils.wandb_utils import presetup_experiment, get_best_result


def perform_experiment():
    config = presetup_experiment()
    callback = CuQuantumCallback(config=config)
    env = get_simulator(config)

    n_steps = int(config['learning']['steps_per_epoch'] / env.num_envs)
    total_timesteps = int(config['train_params']['total_steps'] / env.num_envs)
    model = CuQuantumPPO(policy=GNN_model, env=env, policy_kwargs={'config': config},
                         tensorboard_log='./runs', n_steps=n_steps,
                         batch_size=config['learning']['batch_size'], gamma=config['learning']['gamma'],
                         ent_coef=config['learning']['entropy_weight'], vf_coef=config['learning']['value_weight'],
                         seed=config['train_params']['seed'], device=config['train_params']['device'],
                         n_epochs=config['train_params']['train_epochs_per_rollout'])
    if config['model']['pretrained_model']:
        model = model.load(config['model']['pretrained_model'], env=env, env_check=False,
                           tensorboard_log='./runs', n_steps=n_steps,
                           batch_size=config['learning']['batch_size'], gamma=config['learning']['gamma'],
                           ent_coef=config['learning']['entropy_weight'], vf_coef=config['learning']['value_weight'],
                           seed=config['train_params']['seed'], device=config['train_params']['device'])
        stat_file = os.path.join(os.path.dirname(config['model']['pretrained_model']),
                                 'stats' + os.path.basename(config['model']['pretrained_model'])[5:])
        stats = pickle.load(open(stat_file, 'rb'))
    if config['learning']['epochs'] > 0:
        logging.info('=============Starting Training=======================')
        model.learn(total_timesteps, log_interval=1, callback=callback)
    else:
        assert config['model']['pretrained_model'], "A pretrained model must be loaded in inference mode"
        logging.info('=============Starting Evaluation=======================')
        total_timesteps, callback = model._setup_learn(total_timesteps=0, callback=callback, tb_log_name='PPO')
        callback.on_training_start(locals_=None, globals_=None,
                                   external_constant=stats['reward_normalization_factor'])
        config['eval']['reset_eval_env_after_evaluation'] = False
        config['train_params']['sync_mode'] = 'eval_envs'
        callback.counter = config['eval']['long_eval_freq']
        if config['baselines']['policy_eval_seed'] < 0:
            logging.info('Loading model random generator state')
            if 'numpy_rng_state' in stats:
                np.random.set_state(stats['numpy_rng_state'])
            if 'torch_rng_state' in stats:
                load_torch_rng_state(stats['torch_rng_state'][0], gpu_rng_state=stats['torch_rng_state'][1],
                                     device=model.device)
        else:
            logging.info(f"Setting random seed to {config['baselines']['policy_eval_seed']}")
            np.random.seed(config['baselines']['policy_eval_seed'])
            torch.random.manual_seed(config['baselines']['policy_eval_seed'])
        callback.evaluate_model()
        best_result = callback.update_best_result()
        return best_result


if __name__ == '__main__':
    perform_experiment()
