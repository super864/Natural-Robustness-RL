
"""A training script of PPO on OpenAI Gym Mujoco environments.
This script follows the settings of https://arxiv.org/abs/1709.06560 as much
as possible.
"""
import statistics
import os
import argparse
import functools

import gym
import gym.spaces
import numpy as np
import pandas as pd
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from ppo_MAS import *

import random 
import logging
import sys
import time 
#import cupy as cp 
import csv
import copy


from White_Attacker_PPO import FGSM, PGD

start = time.time()




def compute_grad(action, means, std_devs):
    # compute analytical gradient
    coeff = -(action - means) / ((np.power(std_devs, 3) * (np.sqrt(2 * np.pi))))
    power = -(np.power((action - means), 2)) / (2 * np.power(std_devs, 2))
    exp = np.exp(power)
    grad_a = coeff * exp
    return grad_a



def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )


    parser.add_argument('--budget', type=float, default=1)
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--start_atk', type=int, default='1')
    parser.add_argument('--clip', type=bool, default=True,help='If set to False, actions will be projected based on unclipped nominal action ')
    parser.add_argument("--rollout",default="MAS", type =str,choices = [ "MAS", 'BlackBox',"FGSM", "PGD"], help=" Attack ")
    parser.add_argument("--s",default="l1", type =str,choices = [ "l1", 'l2'], help=" l1 or l2 norm ")

    parser.add_argument("--attack",default="range", type =str,choices = [ "range", "fix",'_'], help="attack method ")   

    parser.add_argument("--attack_steps",default=40, type =int, help=" number of steps use to update the attacker ")

    parser.add_argument("--space",  default= "caction",choices = ["caction", "aspace" ,'_'], help=" caction: percent of current action, aspace: percent of action space")

    parser.add_argument("--percentage", type=int, default= 25 ,help="attack pecentage ") # choices = [25,50,100, 200,0], 

    parser.add_argument("--objective", default= 'action',choices = ["action","obs", "None",'_'], help="attack target")

    parser.add_argument("--direction", default= 'same', choices = ["same", "flip", 'random', 'random_individually','_'], help="attack direction")

    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs


    assert process_seeds.max() < 2 ** 32
    print("process_seeds--------", args.seed)


    args.outdir = experiments.prepare_output_dir(args, args.outdir)







    def make_env(test,process_idx):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        #process_idx = 0
        process_seed = int(process_seeds[process_idx])



        print("process_seeds1----------", args.seed)
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed

        print("test----------", test)
        print("env_seed----------", env_seed)

        env.seed(env_seed)


        #env.render()
        
        #print("env______", env)
        # Cast observations to float32 because our model uses float32
        # env = pfrl.wrappers.CastObservationToFloat32(env)
        # if args.monitor:
        #     env = pfrl.wrappers.Monitor(env, args.outdir)
        # if args.render:
        #     env = pfrl.wrappers.Render(env)
        return env

    # Only for getting timesteps, and obs-action spaces
    sample_env = make_env(test=False, process_idx = 0 )
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    action_size = action_space.low.size
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,

    )



    spy = PPO_Adversary(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,

    )
    # if args.load or args.load_pretrained:
    #     # either load or load_pretrained must be false
    #     assert not args.load or not args.load_pretrained
    #     if args.load:
    #         agent.load(args.load)
    #     else:
    #         agent.load(utils.download_model("PPO", args.env, model_type="final")[0])


    # experiments.train_agent_batch_with_evaluation(
    #     agent=agent,
    #     env=make_batch_env(False),
    #     eval_env=make_batch_env(True),
    #     outdir=args.outdir,
    #     steps=args.steps,
    #     eval_n_steps=None,


    #     eval_n_episodes=args.eval_n_runs,
    #     eval_interval=args.eval_interval,
    #     log_interval=args.log_interval,
    #     max_episode_len=timestep_limit,
    #     save_best_so_far_agent=False,
    # )
    
    
    epsilon = args.epsilon
    lr = args.lr
    budget = args.budget
    update_steps = int(args.attack_steps)

    phi = lambda x: x.astype(np.float32, copy=False)
    if args.load:

        assert not args.load or not args.load_pretrained
        agent.load(args.load)
        #env = make_batch_env(True) 
        env = make_env(True,0)
        if args.render:
            env.render()
        max_episode_len = env.spec.max_episode_steps
        #agent.eval_model()    #process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
        final_rewards = {}

        #print("Fix_attack_of_current_action_same_direction_25%")

        Obs_act_Dataframe  =  pd.DataFrame()
        
        with agent.eval_mode():
            for p in range(100):
                obs = env.reset()
                obs = phi(obs)
                R = 0
                t = 0
                
                step_number = [] 
                obs_value = []
                action_value= []
                rewards_value = []
                while True:
                    # Uncomment to watch the behavior in a GUI window
                    if args.render:
                        env.render()

                                        #print(attack)
                                        
                                        
                    if args.rollout == 'MAS':
                        #print('Running MAS')

                        
                        if t < args.start_atk:
                            
                            
                            
                            action = agent.act(obs)
                            if args.clip is True:
                                action = np.clip(action, -1, 1)
 
                        else:
                            #env.render()
                            action_dist, vs_pred = spy.act_forward(obs)
                            action = spy.act(obs)
                            if args.clip is True:
                                action = np.clip(action, -1, 1)
                            means = []
                            std_devs = []
                            for k in range(len(action_dist.mean.data[0])):
                                means.append(np.array(action_dist.mean[0][k].data))
                                var = np.exp(np.array(action_dist.variance[0][k].data))
                                std_devs.append(np.sqrt(var))
        
                            grad_a = compute_grad(action, means, std_devs)
                            adv_action = action - (lr * grad_a)
        
                            grad_a = compute_grad(adv_action, means, std_devs)
                            adv_action_new = adv_action - (lr * grad_a)
        
                            counter = 0
                            while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                                # print('Optimizing')
                                adv_action = adv_action_new
                                grad_a = compute_grad(adv_action, means, std_devs)
                                adv_action_new = adv_action - (lr * grad_a)
                                counter += 1
        
                            delta = adv_action_new - action
                            if args.s == 'l2':
                                proj_spatial_delta = l2_spatial_project(delta, budget)
                            elif args.s == 'l1':
                                proj_spatial_delta = l1_spatial_project2(delta, budget)
                
                            proj_action = action + proj_spatial_delta
                            action = np.clip(proj_action, -1, 1)

                            
                            #print('test episode:', i, 'R:', R)

                    elif args.rollout == 'FGSM':

                        FGSM_attack = FGSM(agent, obs, epsilon)
                        #obs_critic_attack = FGSM_attack.generate(make_env,phi,gamma=0.99)
                        env_copy = copy.deepcopy(env)
                        obs_actor_attack = FGSM_attack.fgsm_perturbation(env_copy)


                        action = agent.act(obs_actor_attack)



                    elif args.rollout == 'PGD':

                        PGD_attack = PGD(model=agent, obs=obs, gamma = 0.995, num_steps=update_steps, step_size=lr, eps=epsilon,)
                        #obs_critic_attack = FGSM_attack.generate(make_env,phi,gamma=0.99)
                        env_copy = copy.deepcopy(env)
                        obs_actor_attack = PGD_attack.PGD_perturbation(args.clip,env_copy)
                        action = agent.act(obs_actor_attack)
                            
                    elif args.rollout == 'BlackBox':
                        
                        if args.objective == 'action':
                             
    
    
                            if args.attack == "fix":
                                if args.space == 'caction':
                                    if args.direction == 'same' :       
    
                                        attack = 1 + args.percentage * 0.01 
                                        #print('attack', attack)
                                        action = agent.act(obs)
                                        #print('action', action)
                                        action = attack * action
                                        #print('action', action)
    
                                    elif args.direction == 'flip' :
                                        attack = 1 - args.percentage * 0.01 
                                        #print('attack', attack)
                                        action = agent.act(obs)
                                        #print('action', action)
                                        action = attack * action
                                        #print('action', action)
    
                                    elif args.direction == 'random' :
                                        sign = random.choice([-1, 1])
                                        attack = 1 - sign * args.percentage * 0.01 
                                        action = agent.act(obs)
    
                                        action = attack * action
    
    
                                    elif args.direction == 'random_individually' :
                                        action =  agent.act(obs)
                                        #print('action', action)
                                        action = [(1 + args.percentage * 0.01 * random.choice([-1, 1])) * x for x in action]
                                        #print('action', action)
                                    
    
    
                                elif args.space == 'aspace': 
    
                                    if args.direction == 'same' :  
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]     
    
                                        #print('attack', attack)
    
                                        action = agent.act(obs)
    
                                        #print('action', action)
    
                                        action_temp = [] 
    
                                        for i in range(len(action)): 
    
                                            if action[i] < 0 :
                                                a = action[i] - attack[i]
                                                action_temp.append(a)
    
                                            elif action[i] > 0 :
    
                                                b = action[i] + attack[i]
                                                action_temp.append(b)
    
                                        action = np.array(action_temp)
    
                                        #print('action', action)
    
    
    
                                    elif args.direction == 'flip' :
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
    
                                        #print('attack', attack)
    
                                        action = agent.act(obs)
                                        #print('action', action)
    
                                        action_temp = [] 
    
                                        for i in range(len(action)): 
    
                                            if action[i] < 0 :
                                                a = action[i] + attack[i]
                                                action_temp.append(a)
    
                                            elif action[i] > 0 :
    
                                                b = action[i] - attack[i]
                                                action_temp.append(b)
    
                                        action = np.array(action_temp)
    
                                        #print('action', action)
    
    
    
                                    elif args.direction == 'random':
                                        sign = random.choice([-1, 1])
                                        attack = [sign * args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
                                        #print('attack', attack)
    
                                        action = agent.act(obs)
                                        #print('action', action)
                                        action = attack +  action
                                        #print('action', action)
    
    
                                    elif args.direction == 'random_individually' :
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
                                        #print('attack', attack)
                                        new_attack = [ random.choice([-1, 1]) * x for x in attack]
    
                                        #print('new_attack', new_attack)
                                        action =  agent.act(obs) 
    
                                        #print('action', action)
    
                                        action =  action + new_attack
    
                                        print('action', action)
    
                            elif args.attack == "range":
                                range_number = args.percentage * 0.01 
                                #print('range_number', range_number)
                                attack = [random.uniform(-range_number,range_number) for x in range(np.shape(env.action_space)[0])]
                                #print('attack', attack)
                                action = agent.act(obs)
                                #print('action', action)
                                action = action + attack
                                #print('action', action)
    
                        elif args.objective == 'obs':
    
    
                            if args.attack == "fix":
    
                                if args.space == 'caction':
                                    if args.direction == 'same' :  
    
                                        #print("obs_11", obs )
                                        obs = (1+ args.percentage * 0.01 )  * obs 
                                        #print("obs", obs )
                                        obs = phi(obs)
    
    
                                    elif args.direction == 'flip' :
                                        #print("obs_11", obs )
                                        obs = (1 - args.percentage * 0.01 )  * obs 
                                        obs = phi(obs)
                                        #print("obs", obs )
    
                                    elif args.direction == 'random' :
                                        sign = random.choice([-1, 1])
                                        #print("sign", sign )
                                        #print("obs_11", obs )
                                        obs = (1+ sign * args.percentage * 0.01 ) * obs
                                        obs = phi(obs)
                                        #print("obs_11", obs )
    
                                    elif args.direction == 'random_individually' :
    
    
                                        #print("obs_11", obs)
    
                                        obs = np.array([(1 + args.percentage * 0.01 * random.choice([-1, 1]))* x for x in obs])
                                        obs = phi(obs)
                                        # print("obs", type(obs) )
                                        #action = agent.act(obs)
    
                                        #print("obs", obs )
    
    
                                elif args.space == 'aspace': 
    
                                    if args.direction == 'same' : 
                                        
                                        #print('obs11', np.shape(obs))
    
                                        attack_obs = [args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])] 
    
                                        obs_temp = []
    
    
                                        for i in range(len(obs)): 
    
    
                                            if obs[i] < 0 :
                                                a = obs[i] - attack_obs[i]
                                                obs_temp.append(a)
    
                                            elif obs[i] > 0 :
    
                                                b = obs[i] + attack_obs[i]
                                                obs_temp.append(b)
    
                                            elif obs[i] == 0 :
                                                #print("obs[i]--",obs[i])
                                                #print("attack_obs[i]--",attack_obs[i])
    
                                                if args.zeros == 'zero_zero':
                                                    b = obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_zero'
    
                                                elif args.zeros == 'zero_negative':
    
                                                    b = obs[i] - attack_obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_negative'
    
    
                                                elif args.zeros == 'zero_postive':
    
                                                    b = obs[i] + attack_obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_postive'
    
    
                                        #print('zeros', list(obs).count(0))
                                        #print('obs', np.shape(obs))
                                        obs = np.array(obs_temp)
    
                                        obs = phi(obs)
    
    
    
                                                              
    
                                    elif args.direction == 'flip' :
    
                                        
                                        
                                        attack_obs = [args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])] 
    
                                        obs_temp = []
    
                                        #print("len(obs)", len(obs))
                                        #print("count(obs)", list(obs).count(0))
    
                                        for i in range(len(obs)): 
    
                                            if obs[i] < 0 :
                                                #print("obs[i]",obs[i])
                                                #print("attack_obs[i]",attack_obs[i])
                                                a = obs[i] + attack_obs[i]
    
                                                #print("a",a)
                                                obs_temp.append(a)
    
                                            elif obs[i] > 0 :
                                                #print("obs[i]--",obs[i])
                                                #print("attack_obs[i]--",attack_obs[i])
    
                                                b = obs[i] - attack_obs[i]
                                                #print("b",b)
                                                obs_temp.append(b)
    
    
                                            elif obs[i] == 0 :
                                                #print("obs[i]--",obs[i])
                                                #print("attack_obs[i]--",attack_obs[i])
    
                                                b = obs[i]
    
                                                if args.zeros == 'zero_zero':
                                                    b = obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_zero'
    
                                                elif args.zeros == 'zero_negative':
                                                    #print('-----')
    
                                                    b = obs[i] + attack_obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_negative'
    
    
                                                elif args.zeros == 'zero_postive':
    
                                                    b = obs[i] - attack_obs[i]
                                                    #print("b",b)
                                                    obs_temp.append(b)
                                                    name = 'zero_postive'
    
    
                                        # print("type", np.shape(obs))
                                        # print("typew", np.shape(obs_temp))
                                        # print("typee", np.shape(attack_obs))
    
                                        # print("len(obs)", len(obs))
                                        obs = np.array(obs_temp)
                                        obs = phi(obs)
                                        #print("len(obs)", len(obs))
                                        #print("count(obs)", list(obs).count(0))
                                        #obs = obs_temp
    
                                        #print('obs', obs)
    
                                    elif args.direction == 'random' :
                                        sign = random.choice([-1, 1])
    
                                        #print('obs11', obs)
                                        #print('sign', sign)
                                        attack_obs = [sign * args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])]  
                                        #print('attack_obs', attack_obs)
                                        obs = obs + attack_obs   
    
                                        obs = phi(obs)
                                        # print('obs', obs)
    
    
                                        
    
                                    elif args.direction == 'random_individually' :
                                        #print('obs11', obs)
                                        attack_obs = [ args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])]
    
                                        #print('attack_obs', attack_obs)
                                        new_attack_obs = [ random.choice([-1, 1]) * x for x in attack_obs]
    
                                        #print('new_attack_obs', new_attack_obs)
    
    
                                        obs = obs + new_attack_obs  
                                        obs= phi(obs)
    
                                        #print('obs', obs)
    
                            elif args.attack == "range":
    
                                #print('obs11', obs)
                                range_number = args.percentage * 0.01 
                                #print('range_number', range_number)
                                attack_obs = [random.uniform(-range_number,range_number) for x in range(np.shape(env.observation_space)[0])]
                                #print('attack_obs', attack_obs)
    
                                obs = obs + attack_obs
    
                                #print('obs', obs)
                                obs = phi(obs)
    
    
                            
                            action = agent.act(obs)
    
                            zero_number_obs = (len(action),list(action).count(-1))
                        elif args.objective == 'None':
                            action = agent.act(obs)
                        
                        
                    #if args.objective == 'obs' or args.objective == 'None':
                    obs_value.append(obs)
                    action_value.append(action)
                    step_number.append(t)

                    #print("len(action)", len(action))
                    #print("count(action)", list(action).count(0))
                    obs, r, done, _ = env.step(action)
                    obs = phi(obs)
                    R += r
                    rewards_value.append(r)
                    t += 1
                    reset = t == max_episode_len
                    agent.observe(obs, r, done, reset)

                    if done or reset:
                        #if args.objective == 'obs' or args.objective == 'None':
                        episode_name = np.repeat(p, len(step_number))
                        data_frame_temp = pd.DataFrame({"action" : action_value,"obs":obs_value, "rewards": rewards_value,"steps": step_number, "episode":episode_name } )  
                        
                        Obs_act_Dataframe = Obs_act_Dataframe.append(data_frame_temp)
                        
                        break


                final_rewards[str(p)] = R
                
                print('evaluation episode:', p, 'R:', R)
    else:
        steps = args.steps
        max_episode_len = timestep_limit
        episode_r = 0
        episode_idx = 0
        episode_len = 0
        n_updates = 0
        t = 0


        
        env = make_env(False,0)


        obs = env.reset()
        obs = phi(obs)

       # log_interval = 500 




        values_episodes = {}
        # rewards = []
        # episodes = []

        episodes_number = []
        while t <= args.steps:
            if args.render:
                env.render()

            #print("steps", t)

            print("steps", t)
            # random an attack from(-1,1 )

            #print('-----------------------------------:', np.shape(env.action_space)[0])


            attack = [random.uniform(-1,1) for x in range(np.shape(env.action_space)[0])]
            #print( "attack",attack)

            action = agent.act(obs) #+ attack# plus random noise here 
            #print("action", action)


            #print("env.observation_space.high",env.observation_space.high)

            #print("env.observation_space.low",env.observation_space.low)

            #print("env.action_space.high",env.action_space.high ) 
            #print("env.action_space.low",env.action_space.low)
            #print("action type", np.shape(action))
            #print("action steps", t)
            obs, r, done, info = env.step(action)
            obs = phi(obs)
            t += 1


            episode_r += r

            episode_len += 1
            reset = episode_len  == max_episode_len
            agent.observe(obs, r, done, reset)

            # if is_update == True:

            #     n_updates += 1 

            if done or reset:
                print('------------episode:', episode_idx, 'R:', episode_r)

                # rewards.append(episode_r)            
                # episodes.append(episode_idx)

                values_episodes[str(episode_idx)] = episode_r

                episode_r = 0
                episode_idx += 1
                #save_file(args.outdir,agent,env,episode_r,episode_idx)
                episode_len = 0
                obs = env.reset()
                obs = phi(obs)

            if t % 100000 == 0: #( take last 10 episodes rewards)
            #     rewards.append(episode_r)
                episodes_number.append(episode_idx)
            #print("RRRRRR", episode_r)


        final_rewards ={}
        for i in range(len(episodes_number)):
            temp = []
            for y in range(1,11):
                temp.append(values_episodes[str(episodes_number[i]-y)])

            final_rewards[str(episodes_number[i])] = temp


            episodes_number[i]







        #print('rewards.',rewards)

       # print('episode_idx.',episode_idx)
        
        #print('episodes_number.',episodes_number)
        #print('values_episodes.',values_episodes)

        #print('final_rewards.',final_rewards)



        save_dir =  'withoutattack' + args.outdir + '/' + args.env + '_' +  str(args.seed) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "episodes_number.csv"), "w") as f:
            f = csv.writer(f, delimiter = ',')
            f.writerow(["episodes_number"])
            f.writerow([episodes_number])


        with open(os.path.join(save_dir, "values_episodes.csv"), "w") as f:
            f = csv.DictWriter(f, values_episodes.keys())
            f.writeheader()
            f.writerow(values_episodes)


        with open(os.path.join(save_dir, "final_rewards.csv"), "w") as f:
            f = csv.DictWriter(f, final_rewards.keys())
            f.writeheader()
            f.writerow(final_rewards)


            # f.write("episodes_number\n")
            # f.write(str(episodes_number))
            # f.write('\n')
            # f.write("values_episodes\n")
            # f.write(str(values_episodes))
            # f.write('\n')
            # f.write("final_rewards\n")
            # f.write(str(final_rewards))


        agent.save(save_dir +'agent')
        print('done')






    if args.load:


        if args.rollout == 'BlackBox':
        
            if 'name' in locals(): 

                #exit()
                save_folder =  'Testing_Result' + '/' + str(args.objective) + '_' +str(args.attack) + '_attack_' \
                    + str(args.space) + str(args.direction)+ str(args.percentage) + '%' + '/'+ "Zero_Pixels"+ '/'+ 'Result' + '_test_' + str(args.env)[:-3] +"_" +str(name)+ '/'
                    
                    
                save_result =  'Testing_Attack'+'_'+ str(args.objective)+"_" + str(args.attack)  +"_"+str(args.env)+'_'+ str(args.seed)+ '.csv'
        
                
                if args.objective == "obs" or args.objective == 'None':
                    save_datframe_folder =  'Obs_act_relation' + '/' + str(args.objective) + '_' +str(args.attack) + '_attack_' \
                        + str(args.space) + str(args.direction)+ str(args.percentage) + '%' + '/'+"Zero_Pixels"+ '/' 'Result' + '_test_' + str(args.env)[:-3] +'_ '+ str(name) +  '/'
                        
                    save_dataframe_result =   str(args.objective)+"_" + str(args.attack)  +"_"+str(args.env)[:-3]+'_'+str(name)+'_'+ str(args.seed)+ '.csv'
        
        
        
        
        
            else:

                # exit()
                save_folder =  'Testing_Result' + '/' + str(args.objective) + '_' +str(args.attack) + '_attack_' \
                    + str(args.space) + str(args.direction)+ str(args.percentage) + '%' + '/'+ 'Result' + '_test_' + str(args.env)[:-3] +'/'
                    
                    
                save_result =  'Testing_Attack'+'_'+ str(args.objective)+"_" + str(args.attack)  +"_"+str(args.env)+'_'+ str(args.seed)+ '.csv'
        
                
                if args.objective == "obs" or args.objective == 'None':
                    save_folder =  'Obs_act_relation' + '/' + str(args.objective) + '_' +str(args.attack) + '_attack_' \
                        + str(args.space) + str(args.direction)+ str(args.percentage) + '%' + '/'+ 'Result' + '_test_' + str(args.env)[:-3] + '/'
                        
                    save_result =   str(args.objective)+"_" + str(args.attack)  +"_"+str(args.env)[:-3]+'_' + str(args.seed)+ '.csv'
        
        
        
            adr = os. getcwd()+ '/'
                
                
            store_data( df =Obs_act_Dataframe, fn=adr + os.path.join(save_datframe_folder,save_dataframe_result ), compression='brotli' )

            
                
                    
        elif args.rollout == 'MAS':
            
            save_folder =  'MAS_Attack' + '/' + "Starting_Point_" + str(args.start_atk) + '_' + "Clip_"+str(args.clip ) + '_Learning_rate_' +str(args.lr ) +\
                '_epsilon_' + str(args.epsilon)+ '_budget_' + str(args.budget)+ '_norm_' + str(args.s)+'/'+ str(args.env)[:-3] +'/'
                
                
            save_result =  'Seed' + str(args.seed)+ '.parquet'
            
            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        
            adr = os. getcwd()+ '/'
                
                
            store_data( df =Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli' )

    
        elif args.rollout == 'FGSM':



            save_folder =  'FGSM_Attack' + '/' + 'Epsilon_' + str(args.epsilon) +'/'+ str(args.env)[:-3] +'/'
                
                
            save_result =  'Seed' + str(args.seed)+ '.parquet'
            
            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        
            adr = os. getcwd()+ '/'
                
                
            store_data( df =Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli')


        elif args.rollout == 'PGD':



            save_folder =  'PGD_Attack' + '/' + 'Epsilon_' + str(args.epsilon)+'_Learning_rate_' +str(args.lr ) + "_Update_steps_" +str(update_steps ) +'/'+ str(args.env)[:-3] +'/'
                
                
            save_result =  'Seed' + str(args.seed)+ '.parquet'
            
            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        
            adr = os. getcwd()+ '/'
                
                
            store_data( df =Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli')


    # else:
    #     save_folder =  'Training_Result'  + str(args.env) + '/' 
    #     save_result =  'Training_Result'+"_"+str(args.env)+'_' + str(args.seed) + '.csv'
    # #save_result =  'final_rewards' + str(args.seed) + '.csv'
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # with open(os.path.join(save_folder,save_result ), "w") as f:
    #     f = csv.DictWriter(f, final_rewards.keys())
    #     f.writeheader()s
    #     f.writerow(final_rewards)

        
        
        
def store_data(df, fn, compression=''):


    print('writing...', end='', flush=True)
    
    compression = 'brotli' if compression == '' else compression
    
    
    df.to_parquet(fn, engine='pyarrow', compression=compression)
    
    
    
    
if __name__ == "__main__":
    main()
    end = time.time()
    print('Total time cost{}'.format(end- start))


# def save_file( outdir, agent, env, episode_r,episode_idx ):
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)

#     with open(os.path.join(outdir, "scores.txt"), "w") as f:
#         f.write(str(agent))
#         f.write(str(env))
#         f.write(str(episode_r))
#         f.write(str(episode_idx))


# def write_header(outdir, agent, env):
#     # Columns that describe information about an experiment.
#     basic_columns = (
#         "steps",  # number of time steps taken (= number of actions taken)
#         "episodes",  # number of episodes finished
#         "elapsed",  # time elapsed so far (seconds)
#         "mean",  # mean of returns of evaluation runs
#         "median",  # median of returns of evaluation runs
#         "stdev",  # stdev of returns of evaluation runs
#         "max",  # maximum value of returns of evaluation runs
#         "min",  # minimum value of returns of evaluation runs
#     )
#     with open(os.path.join(outdir, "scores.txt"), "w") as f:
#         custom_columns = tuple(t[0] for t in agent.get_statistics())
#         env_get_stats = getattr(env, "get_statistics", lambda: [])
#         assert callable(env_get_stats)
#         custom_env_columns = tuple(t[0] for t in env_get_stats())
#         column_names = basic_columns + custom_columns + custom_env_columns
#         print("\t".join(column_names), file=f)



    # save_dir = 'results/' + args.outdir + '/'
    # if not os.path.exists(args.outdir):
    #     os.makedirs(args.outdir)

    # agent.save(args.outdir +'agent')
    # print('done')


    # logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    # pfrl.experiments.train_agent_with_evaluation(
    # agent,
    # env,
    # steps=args.steps,           # Train the agent for 2000 steps
    # eval_n_steps=None,       # We evaluate for episodes, not time
    # eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    # train_max_episode_len = max_episode_len,  # Maximum length of each episode
    # eval_interval= args.eval_interval,   # Evaluate the agent after every 1000 steps
    # outdir='args.outdir',      # Save everything to 'result' directory
    # )

    # with agent.eval_mode():
    # for i in range(10):
    #     obs = env.reset()
    #     R = 0
    #     t = 0
    #     while True:
    #         # Uncomment to watch the behavior in a GUI window
    #         # env.render()
    #         action = agent.act(obs)
    #         obs, r, done, _ = env.step(action)
    #         R += r
    #         t += 1
    #         reset = t == 200
    #         agent.observe(obs, r, done, reset)
    #         if done or reset:
    #             break
    #     print('evaluation episode:', i, 'R:', R)

