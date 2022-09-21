import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch.nn import functional as F

import pfrl
from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.agents.td3 import *

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

class TD3_Adversary(TD3):
    """Twin Delayed Deep Deterministic Policy Gradients (TD3).
    See http://arxiv.org/abs/1802.09477
    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        policy_update_delay (int): Delay of policy updates. Policy is updated
            once in `policy_update_delay` times of Q-function updates.
        target_policy_smoothing_func (callable): Callable that takes a batch of
            actions as input and outputs a noisy version of it. It is used for
            target policy smoothing when computing target Q-values.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_policy",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
    )

    def __init__(
        self,
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        n_times_update=1,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        policy_update_delay=2,
        target_policy_smoothing_func=default_target_policy_smoothing_func,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.q_func1.to(self.device)
            self.q_func2.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.policy_update_delay = policy_update_delay
        self.target_policy_smoothing_func = target_policy_smoothing_func

        self.t = 0
        self.policy_n_updates = 0
        self.q_func_n_updates = 0
        self.last_state = None
        self.last_action = None

        # Target model
        self.target_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)
        self.policy_loss_record = collections.deque(maxlen=100)
        
        
        
        
    def act_forward(self, batch_obs, adv_action):

        action = self.batch_select_onpolicy_action(batch_obs)
        action = np.array(action)
        
        q1 = torch.flatten(self.q_func1((torch.from_numpy(batch_obs), torch.from_numpy(action))))
        q2 = torch.flatten(self.q_func2((torch.from_numpy(batch_obs), torch.from_numpy(action))))
        q = torch.min(q1, q2)
    
        
    
        q1_adv = torch.flatten(self.q_func1((torch.from_numpy(batch_obs), adv_action)))
        q2_adv = torch.flatten(self.q_func2((torch.from_numpy(batch_obs), adv_action)))
        q_adv = torch.min(q1_adv, q2_adv)
    
        return action,q, q_adv

    