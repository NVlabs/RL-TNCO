import logging
import os

config = dict()

network = dict(
        n_nodes=400,
        n_edges=800,
        mean_connectivity=3,
        tensor_dim=6,
        train_files=None,
        test_files='datasets/Sycamore/circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p',
        type='random'
        )

model = {
    'score_to_prob'            : 'probs',  # logits or probs or probs_unnormalized
    'eps'                      : 10 ** -2,
    'num_gnn_layers'           : 4,
    'num_intermediate_features': 128,
    'use_batch_norm'           : False,
    'use_pair_norm'            : True,
    'agg_type'                 : 'mean',
    'use_bias'                 : False,
    'pretrained_model'         : None,
    }

train_params = {
    'seed'                                    : 127,
    'save_freq'                               : 500,
    'normalize_advantage'                     : False,
    'reward_function'                         : 'flops_only',
    'device'                                  : 'auto',
    'n_envs'                                  : 8,
    'dummy_parallelism'                       : True,
    'total_steps'                             : 1004000,
    'edge_normalization'                      : 'robust_features',  # None, column_max, global_max
    'external_reward_normalization'           : 1e17,
    'precalc_solver'                          : 'ctg_kahypar',  # 'ctg_kahypar', 'oe_greedy'
    'baseline_for_action_elimination_in_eval' : 'oe_greedy',
    'baseline_for_action_elimination_in_train': 'oe_greedy',
    'total_return_momentum'                   : 0.5,
    'use_multi_GPU'                           : False,
    'block_decreasing_action'                 : True,
    'block_decreasing_action_train'           : False,
    'add_feasible_actions_as_features'        : False,
    'limit_to_baseline_improving_actions'     : True,
    'sync_mode'                               : 'eval_envs',  # 'all',  # 'train',  # 'all'
    'clip_reward'                             : 1e6,
    'clip_return'                             : 1e6,
    'log_scale_greedy_score'                  : True,
    'max_number_of_fixed_eqs_per_env'         : 1000,
    'additional_rescaling_factor'             : 1,
    'train_epochs_per_rollout'                : 8,
    'suffix_solver'                           : None,  # 'greedy',
    'max_prefix'                              : 5
    }

representation = {
    'node_features'   : 0,
    'edge_features'   : 2,
    'global_features' : 2,
    'add_greedy_score': True,
    'greedy_weight'   : 10
    }

logger = dict(
        logPath='',
        logfileName='log.txt',
        logging_level=logging.INFO)

learning = dict(
        steps_per_epoch=4000,
        batch_size=500,
        epochs=800,
        gamma=1.0,
        pi_lr=3e-3,
        train_v_iters=10,
        lam=0.97,
        entropy_weight=0,
        value_weight=0.001,
        normalize_advantage=False,
        use_optimistic_buffer=True,
        update_policy_using_optimistic_buffer=True,
        update_v_using_optimistic_buffer=True,
        optimistic_epochs=8,
        optimistic_buffer_size=4096,
        convex_factor_for_adv_norm=0.5,
        v_norm=2,
        )

eval = dict(evaluation=True,
            reset_eval_env_after_evaluation=True,
            n_simultaneous_path_per_eq=2,
            long_eval_freq=1000,
            eval_freq=500,
            long_eval_episodes=500,
            eval_seed=0,
            max_eval_envs=200)

baselines = dict(num_eqs=100,
                 num_nodes=25,
                 depth=10,
                 time_limit=6000,
                 baseline='cuquantum',
                 cutensor_samples=100,
                 baseline_repeats=10,
                 split=None,
                 policy_eval_seed=-1)

wandb = {"project": "RL-TNCO"}

default_config = dict(
        learning=learning,
        logger=logger,
        representation=representation,
        train_params=train_params,
        model=model,
        network=network,
        # visualization=visualization,
        wandb=wandb,
        eval=eval,
        baselines=baselines
        )


def get_config():
    return default_config


def create_logger(config, log_to_file=False):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]  %(message)s")
    rootLogger = logging.getLogger()

    if log_to_file:
        fileHandler = logging.FileHandler(os.path.join(config['logger']['logPath'], config['logger']['logfileName']))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(config['logger']['logging_level'])
    return rootLogger
