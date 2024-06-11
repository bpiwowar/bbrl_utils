from typing import List, Union
import numpy as np
import torch.nn as nn
from itertools import chain
from bbrl import get_arguments, get_class


def setup_optimizer(cfg_optimizer, *agents: Union[nn.Module, nn.Parameter]):
    """Setup an optimizer for a list of agents"""
    optimizer_args = get_arguments(cfg_optimizer)
    parameters = [
        agent.parameters() if isinstance(agent, nn.Module) else [agent]
        for agent in agents
    ]
    optimizer = get_class(cfg_optimizer)(chain(*parameters), **optimizer_args)
    return optimizer


def copy_parameters(source: nn.Module, target: nn.Module):
    """Copy parameters from a model a to target"""
    for source_p, target_p in zip(source.parameters(), target.parameters()):
        target_p.data.copy_(source_p)


def soft_update_params(net, target_net, tau):
    r"""Soft parameter updates

    To update the target critic, one uses the following equation: $\theta'
    \leftarrow \tau \theta + (1- \tau) \theta'$ where $\theta$ is the vector of
    parameters of the critic, and $\theta'$ is the vector of parameters of the
    target critic. The `soft_update_params(...)` function is in charge of
    performing this soft update.


    :param net: The source module
    :param target_net: The target module
    :param tau: Weight for old parameters should be kept (0 to 1)
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Function used for orthogonal initialization of the layers. Taken from here
    in the cleanRL library:
    https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_ortho_mlp(sizes: List[int], activation, output_activation=nn.Identity()):
    r"""Helper function to build a multi-layer perceptron

    function from $\mathbb R^n$ to $\mathbb R^p$ with orthogonal initialization

    :param sizes: the number of neurons at each layer
    :param activation: a PyTorch activation function (after each
        layer but the last)
    :param output_activation: a PyTorch activation function (last
        layer)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), act]
    return nn.Sequential(*layers)


def build_mlp(sizes: List[int], activation: nn.Module, output_activation=nn.Identity()):
    r"""Helper function to build a multi-layer perceptron

    The function below builds a multi-layer perceptron where the size of each
    layer is given in the `size` list. We also specify the activation function of
    neurons at each layer and optionally a different activation function for the
    final layer.

    The output is a function/module from $\mathbb R^n$ to $\mathbb R^p$


    :param sizes: the number of neurons at each layer
    :param activation: a PyTorch activation function (after each layer but the last)
    :param output_activation: a PyTorch activation function (last layer)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)
