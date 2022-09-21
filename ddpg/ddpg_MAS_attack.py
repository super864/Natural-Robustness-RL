import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters

from pfrl.agents.ddpg import *
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





class DDPG_Adversary(DDPG):
    """Deep Deterministic Policy Gradients.
    This can be used as SVG(0) by specifying a Gaussian policy instead of a
    deterministic policy.
    Args:
        policy (torch.nn.Module): Policy
        q_func (torch.nn.Module): Q-function
        actor_optimizer (Optimizer): Optimizer setup with the policy
        critic_optimizer (Optimizer): Optimizer setup with the Q-function
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
    """

    saved_attributes = ("model", "target_model", "actor_optimizer", "critic_optimizer")

    def __init__(
        self,
        policy,
        q_func,
        actor_optimizer,
        critic_optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=50000,
        minibatch_size=32,
        update_interval=1,
        target_update_interval=10000,
        phi=lambda x: x,
        target_update_method="hard",
        soft_update_tau=1e-2,
        n_times_update=1,
        recurrent=False,
        episodic_update_len=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
    ):

        self.model = nn.ModuleList([policy, q_func])
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.recurrent = recurrent
        assert not self.recurrent, "recurrent=True is not yet implemented"
        if self.recurrent:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=recurrent,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.q_record = collections.deque(maxlen=1000)
        self.actor_loss_record = collections.deque(maxlen=100)
        self.critic_loss_record = collections.deque(maxlen=100)
        self.n_updates = 0

        # Aliases for convenience
        self.policy, self.q_function = self.model
        self.target_policy, self.target_q_function = self.target_model

        self.sync_target_network()
    

            
    def act_forward(self, batch_obs, adv_action):

        action = self._batch_select_greedy_actions(batch_obs)

# =============================================================================
#         transitions = self.replay_buffer.sample(1)
#         
#         onpolicy_actions = self.policy(batch_obs)#.rsample()
# =============================================================================
        
        q = self.q_function((torch.from_numpy(batch_obs), torch.from_numpy(action)))
        
        #print("___________________________")
        
        
        q_adv = self.q_function((torch.from_numpy(batch_obs), adv_action))
        
        return action,q, q_adv
    
    
