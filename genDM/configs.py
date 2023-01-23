# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from util.utils import ConfigDict
from util.scms import SCM
import numpy as np


def handle_config(config: ConfigDict, scm: SCM) -> ConfigDict:
    variables = scm.dag.get_obs_parents(scm.treatment) + [scm.treatment]
    dims = [np.arange(np.sum(scm.dag.var_dims[:i]), np.sum(scm.dag.var_dims[:(i + 1)])) for i in variables]
    config.gc.batch_condition_dims = np.array(dims[:-1]).flatten()
    config.gc.batch_decision_dims = np.array(dims[-1]).flatten()
    config.dmc.decision_dim = scm.dag.var_dims[scm.treatment]
    config.dmc.condition_dim = sum([scm.dag.var_dims[p] for p in scm.dag.get_obs_parents(scm.treatment)])
    config.mle_config.condition_dim = config.dmc.condition_dim
    config.mle_config.decision_dim = config.dmc.decision_dim

    config.dmc.features = config.dmc.features + [config.dmc.decision_dim]
    config.mle_config.features = config.mle_config.features + [config.dmc.decision_dim]
    config.amc.features = config.amc.features + [1]  # output is the ratio
    config.amc.input_dim = sum(scm.dag.var_dims)

    return config


train_config = ConfigDict(
    {
        'gc': ConfigDict(
            {
                'dataset': 'linear_backdoor',
                'seed': 284747,
                'n_samples': 10000,
                'lagrange_lr': 0.5,
                'epochs': 6,
                'MLE_epochs': 100,
                'batch_size': 1000,
                'alpha': 0.1,
                'max_lambda': 1000,
                'inner_steps': 2000,
                'batch_condition_dims': None,
                'batch_decision_dims': None,
                'debug': True,
                'neumann_steps': 10,
                'neumann_lr': 1e-4
            }
        ),
        'dmc': ConfigDict(
            {
                'features': [32, 32],  # without output
                'activation': 'relu',
                'condition_dim': None,
                'decision_dim': None,
                'variance': 1
            }
        ),
        'mle_config': ConfigDict(
            {
                'features': [32, 32],  # without output
                'activation': 'relu',
                'condition_dim': None,
                'decision_dim': None,
                'variance': None
            }
        ),
        'amc': ConfigDict(
            {
                'features': [32, 32],  # without output
                'input_dim': None,
                'activation': 'tanh',
                'momentum': 0.9,
                'stable_eps': 0.
            }
        ),

        'doc': ConfigDict(
            {
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'grad_clip': 1,
            }
        ),

        'aoc': ConfigDict(
            {
                'learning_rate': 0.01,
                'optimizer': 'adam',
                'weight_decay': None,
                'grad_clip': 5,
                # 'momentums': [1., 1.]
            }
        )
    }
)