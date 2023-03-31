import time

import pickle
import numpy as np

from TNCO_solver import TNCOsolver
from utils.wandb_utils import presetup_experiment
from utils.main_utils import read_data_file


def benchmark():
    config = presetup_experiment()
    train_files, eval_file = read_data_file(config)
    with open(eval_file, 'rb') as f:
        eqs, baseline_solutions, _ = pickle.load(f)
    if not isinstance(eqs, list):
        eqs = [eqs]
    operands = []
    for eq in eqs:
        operands.append([np.empty(s) for s in eq[1]])
    tnco_solver = TNCOsolver(config)
    path = tnco_solver.find_path()

if __name__ == '__main__':
    benchmark()
