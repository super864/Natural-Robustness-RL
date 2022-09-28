import collections
import itertools
import random

import numpy as np
from numpy import linalg as LA
import torch
import torch.nn.functional as F

import pfrl
from pfrl import agent
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)
from pfrl.agents import a3c
from pfrl.agents.ppo import *


def l1_spatial_project2(y_orig, budget):

    y_abs = list(map(abs, y_orig))
    u = sorted(y_abs, reverse=True)
    binK = 1
    K = 1
    bin_list = [0] * len(u)
    for i in range(1, len(u) + 1):
        if (sum([u[r] for r in range(i)]) - budget) / i < u[i - 1]:
            bin_list[i - 1] = binK
            binK += 1

    if sum(bin_list) > 0:
        K = np.argmax(bin_list) + 1

    tau = (sum([u[i] for i in range(K)]) - budget) / K
    xn = [max(item - tau, 0) for item in y_abs]
    l1_norm_y = np.linalg.norm(y_orig, 1)
    for i in range(len(y_orig)):
        if l1_norm_y > budget:
            y_orig[i] = np.sign(y_orig[i]) * xn[i]
    return y_orig

def l2_spatial_project(x, distance):
    norm = l2_spatial_norm(x)
    # print('x',x)
    # print('l2 norm', diff)
    # print('dist',distance)
    if norm <= distance:
        delta = x
    else:
        delta = (x / norm) * distance
    return delta

def l2_spatial_norm(x):
    return LA.norm(x, 2)

class PPO_Adversary(PPO):
    """Proximal Policy Optimization
    See https://arxiv.org/abs/1707.06347
    Args:
        model (torch.nn.Module): Model to train (including recurrent models)
            state s  |->  (pi(s, _), v(s))
        optimizer (torch.optim.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and update in a recurrent
            manner.
        max_recurrent_sequence_len (int): Maximum length of consecutive
            sequences of transitions in a minibatch for updating the model.
            This value is used only when `recurrent` is True. A smaller value
            will encourage a minibatch to contain more and shorter sequences.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.
    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
        n_updates: Number of model updates so far.
        explained_variance: Explained variance computed from the last batch.
    """

    saved_attributes = ("model", "optimizer", "obs_normalizer")

    def __init__(
        self,
        model,
        optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        value_func_coef=1.0,
        entropy_coef=0.01,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        value_loss_stats_window=100,
        policy_loss_stats_window=100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm

        # Contains episodes used for next update iteration  
        self.memory = []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None
        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(maxlen=policy_loss_stats_window)
        self.explained_variance = np.nan
        self.n_updates = 0


    def act_forward(self, batch_obs):
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)
        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            action_distrib, batch_value = self.model(b_state)
            batch_action = action_distrib.sample().cpu().numpy()

        return action_distrib, batch_value
    
    
