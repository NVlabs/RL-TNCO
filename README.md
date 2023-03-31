# RL-TNCO

RL-TNCO is a reinforcement learning algorithm for solving the tensor network (TN) contraction problem. The goal of this algorithm is to find a good contraction order.

You can read about it in more details in our [paper](https://proceedings.mlr.press/v162/meirom22a/meirom22a.pdf). To learn more about tensor network, consider reviewing [this](https://docs.nvidia.com/cuda/cuquantum/cutensornet/overview.html#introduction-to-tensor-networks). 
For a gentle introduction on reinforcement learning, we recommended these [resources](https://stable-baselines3.readthedocs.io/en/master/guide/rl.html).

## Prerequisites

An environment can be set using either Docker or Conda.

1. Docker: use the  `Dockerfile` in the repository.
2. Conda: Create a conda environment, `conda env create -n rl-tnco --file environment.yml`

## Training

The code is based on Stable-Baselines 3 framework.

Training is done by running `main.py`. Parameters can be adjusted by modifying `config.py`. RL-TNCO can be either trained on a predefined tensor network dataset or generate its own random tensor network dataset, based on [opt-einsum](https://optimized-einsum.readthedocs.io/en/stable/) tensor network generator. To train on a specific dataset, set the `train-files` parameter in the config file to the datafile, otherwise set it to `None`. Inference should be performed on a predefined test file.

## Inference

Set the pretrained model parameter in `config.py` to the pretrained model file. We support two methods for performing inference.

1. Use TNCOsolver directly (recommended): See `benchmarking.py` for an example.
2. Using the training pipeline: Reduce the number of training epochs to 0 (`epochs` parameter in `config.py`) and run `main.py`.


## File format

Train (and test file) are pickle files. Each file contains three fields: 

1. Eqs: List of $n$ equations. Each equation is a tuple of length 3, and follows the output of opt-einsum's `helpers.rand_equation()` function. The first element is an equation string in [opt-einsum notation](https://optimized-einsum.readthedocs.io/en/stable/input_format.html) (e.g., "ijk,jkl-\>li"). The second element is a `shape` variable, specifying the dimension for each tensor (e.g., [(2,2,3),(2,3,4)]). The last entry is `size_dict`, specifying the extent of each index.

2. `baseline_solutions` (optional, recommended): A dictionary where each key is the baseline label (e.g., oe-greedy). Each value is a list of $n$ tuples representing the output paths found by the baseline. The tuples entries are: contraction cost, path finding compute time, [PathInfo](https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.PathInfo.html), path as a list of contracted pairs.
This variable is mandatory for path pruning (see paper) during training. We recommended providing this baseline even with a weak baseline. It is not required for inference.
4. `info` (optimal): additional information

## Practical Guidelines

First, set the following parameters according to the instruction in the table.

| **Parameter** | **Interpretation** |
| --- | --- |
| n\_nodes | This should be equal or higher than the number of nodes in the largest TN. If no datafile is provided, this parameter set the number of nodes in the random TNs generated during training. |
| n\_edges | This should be equal or higher than the number of edges in the largest TN. |
| external\_reward\_normalization | A scaling factor. Should be set to the order of magnitude of the cost (#flops). It can be estimated using a fast (and suboptimal) solver. |
| batch\_size | Set it to as high as possible with getting OutOfMemory errors. This value will depend on the TN size. |

Obtaining optimal results may require some tuning. Try and adjust the parameters `value_weight`, `greedy_weight`, and `external_reward_normalization`. It is recommended to use multiple seeds. Feel free to contact us for advice and further tuning instructions depending on your use case.

## Contact us

[Contract us](mailto:emeirom@nvidia.com) for specific pretrained models and questions.
