# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from collections import OrderedDict
from itertools import chain
from typing import List
import numpy as np


class DAG:
    r"""Describes the DAG structure, the variables in the DAG, and their dimensions"""

    def __init__(self, name: str, graph: OrderedDict, latent_dims: List[int], var_dims: List[int]):
        self.name = name
        self.graph = graph  # negative values indicate latent variables
        self.var_dims = np.array(var_dims)  # dimension of each variable
        self.latent_dims = latent_dims
        self.n_latent = sum(n < 0 for n in list(set(chain(*graph.values()))))

    def __str__(self):
        return self.name

    def get_obs_parents(self, var: int) -> List[int]:
        return [i for i in self.graph[var] if i >= 0]


# T ----> Y
def gen_2d(var_dims: List[int]):
    name = '2d'
    graph = OrderedDict([
        (0, []),  # T
        (1, [0, -1]),  # Y
    ])
    latent_dims = [1]  # The dimension of latent noise variables
    return DAG(name, graph, latent_dims, var_dims)


# X -> T -> Y. Identifiable
def gen_backdoor(var_dims: List[int]):
    name = 'backdoor'
    graph = OrderedDict([
        (0, []),  # X
        (1, [0, -1]),  # T
        (2, [0, 1, -2]),  # Y
    ])
    latent_dims = [1, 1]  # The dimension of latent noise variables
    return DAG(name, graph, latent_dims, var_dims)


#      - - - -
#    /        \
#   T -> M -> Y     Identifiable
def gen_frontdoor(var_dims: List[int]):
    name = 'frontdoor'
    graph = OrderedDict([
        (0, [-1, -2]),
        (1, [0, -3]),
        (2, [1, -1, -4]),
    ])
    latent_dims = [1, 1, 1, 1]
    return DAG(name, graph, latent_dims, var_dims)


#           --
#         /   \
#   X -> T -> Y
def gen_iv(var_dims: List[int]):
    name = 'iv'
    graph = OrderedDict([
        (0, []),
        (1, [0, -1, -2]),
        (2, [1, -2, -3]),
    ])
    latent_dims = [1, 1, 1]
    return DAG(name, graph, latent_dims, var_dims)


#       -------
#      /    -  \
#    /     / \ \
#   T -> X ->  Y    Non-identifiable
def gen_leaky(var_dims: List[int]):
    name = 'leaky'
    graph = OrderedDict([
        (0, [-1, -2]),
        (1, [0, -3, -4]),
        (2, [1, -1, -3, -5])
    ])

    latent_dims = [1, 1, 1, 1, 1]
    return DAG(name, graph, latent_dims, var_dims)


# Non-identifiable
def gen_bow(var_dims: List[int]):
    name = 'bow'
    graph = OrderedDict([
        (0, [-1, -3]),
        (1, [0, -1, -2]),
    ])
    latent_dims = [1, 1, 1]
    return DAG(name, graph, latent_dims, var_dims)


def gen_dags(key: str, var_dims: List[int]):
    return {
        'backdoor': gen_backdoor,
        'frontdoor': gen_frontdoor,
        'bow': gen_bow,
        'leaky': gen_leaky,
        'iv': gen_iv,
        '2d': gen_2d
    }[key](var_dims)
