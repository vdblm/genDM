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

    config.dmc.features = [config.dmc.condition_dim] + config.dmc.features + [config.dmc.decision_dim]
    config.amc.features = [sum(scm.dag.var_dims)] + config.amc.features + [1]  # output is the ratio

    return config


train_config = ConfigDict(
    {
        'gc': ConfigDict(
            {
                'dataset': 'linear_backdoor',
                'seed': 284747,
                'n_samples': 5000,
                'lagrange_lr': 0.1,
                'epochs': 100,
                'MLE_epochs': 100,
                'batch_size': 1024,
                'alpha': 0.01,
                'max_lambda': 10.0,
                'inner_steps': 10,
                'batch_condition_dims': None,
                'batch_decision_dims': None
            }
        ),
        'dmc': ConfigDict(
            {
                'features': [32, 32],  # without input/output
                'activation': 'relu',
                'condition_dim': None,
                'decision_dim': None
            }
        ),
        'amc': ConfigDict(
            {
                'features': [32, 32],  # without input/output
                'activation': 'relu',
                'momentum': 0.9,
                'stable_eps': 1e-5
            }
        ),

        'doc': ConfigDict(
            {
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        ),

        'aoc': ConfigDict(
            {
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        )

    }
)
