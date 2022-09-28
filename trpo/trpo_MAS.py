import collections
import random
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pfrl
from pfrl import agent
from pfrl.agents.ppo import (  # NOQA
    _compute_explained_variance,
    _make_dataset,
    _make_dataset_recurrent,
    _yield_minibatches,
    _yield_subset_of_sequences_with_fixed_number_of_items,
)
from pfrl.utils import clip_l2_grad_norm_
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


from pfrl.agents.trpo import *

from numpy import linalg as LA

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



class TRPO_Adversary(TRPO):
    """Trust Region Policy Optimization.
    A given stochastic policy is optimized by the TRPO algorithm. A given
    value function is also trained to predict by the TD(lambda) algorithm and
    used for Generalized Advantage Estimation (GAE).
    Since the policy is optimized via the conjugate gradient method and line
    search while the value function is optimized via SGD, these two models
    should be separate.
    Since TRPO requires second-order derivatives to compute Hessian-vector
    products, your policy must contain only functions that support second-order
    derivatives.
    See https://arxiv.org/abs/1502.05477 for TRPO.
    See https://arxiv.org/abs/1506.02438 for GAE.
    Args:
        policy (Policy): Stochastic policy. Its forward computation must
            contain only functions that support second-order derivatives.
            Recurrent models are not supported.
        vf (ValueFunction): Value function. Recurrent models are not supported.
        vf_optimizer (torch.optim.Optimizer): Optimizer for the value function.
        obs_normalizer (pfrl.nn.EmpiricalNormalization or None):
            If set to pfrl.nn.EmpiricalNormalization, it is used to
            normalize observations based on the empirical mean and standard
            deviation of observations. These statistics are updated after
            computing advantages and target values and before updating the
            policy and the value function.
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Interval steps of TRPO iterations. Every time after
            this amount of steps, this agent updates the policy and the value
            function using data from these steps.
        vf_epochs (int): Number of epochs for which the value function is
            trained on each TRPO iteration.
        vf_batch_size (int): Batch size of SGD for the value function.
        standardize_advantages (bool): Use standardized advantages on updates
        line_search_max_backtrack (int): Maximum number of backtracking in line
            search to tune step sizes of policy updates.
        conjugate_gradient_max_iter (int): Maximum number of iterations in
            the conjugate gradient method.
        conjugate_gradient_damping (float): Damping factor used in the
            conjugate gradient method.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        kl_stats_window (int): Window size used to compute statistics
            of KL divergence between old and new policies.
        policy_step_size_stats_window (int): Window size used to compute
            statistics of step sizes of policy updates.
    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated after `act` or `batch_act` methods are called in the
            training mode.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated after `act` or `batch_act`
            methods are called in the training mode.
        average_kl: Average of KL divergence between old and new policies.
            It's updated after the policy is updated.
        average_policy_step_size: Average of step sizes of policy updates
            It's updated after the policy is updated.
    """

    saved_attributes = ("policy", "vf", "vf_optimizer", "obs_normalizer")

    def __init__(
        self,
        policy,
        vf,
        vf_optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        entropy_coef=0.01,
        update_interval=2048,
        max_kl=0.01,
        vf_epochs=3,
        vf_batch_size=64,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        line_search_max_backtrack=10,
        conjugate_gradient_max_iter=10,
        conjugate_gradient_damping=1e-2,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        kl_stats_window=100,
        policy_step_size_stats_window=100,
        logger=getLogger(__name__),
    ):

        self.policy = policy
        self.vf = vf
        self.vf_optimizer = vf_optimizer
        self.obs_normalizer = obs_normalizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.max_kl = max_kl
        self.vf_epochs = vf_epochs
        self.vf_batch_size = vf_batch_size
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.line_search_max_backtrack = line_search_max_backtrack
        self.conjugate_gradient_max_iter = conjugate_gradient_max_iter
        self.conjugate_gradient_damping = conjugate_gradient_damping
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.logger = logger

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.vf.to(self.device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
        else:
            self.device = torch.device("cpu")

        if recurrent:
            self.model = pfrl.nn.RecurrentBranched(policy, vf)
        else:
            self.model = pfrl.nn.Branched(policy, vf)

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.kl_record = collections.deque(maxlen=kl_stats_window)
        self.policy_step_size_record = collections.deque(
            maxlen=policy_step_size_stats_window
        )
        self.explained_variance = np.nan

        self.last_state = None
        self.last_action = None

        # Contains episodes used for next update iteration
        self.memory = []
        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None
        

    
    def act_forward(self, batch_obs):
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            
            action_distrib, batch_value = self.model(b_state)
            batch_action = action_distrib.sample().cpu().numpy()


        return action_distrib, batch_value
    