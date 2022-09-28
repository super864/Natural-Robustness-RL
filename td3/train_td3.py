"""A training script of TD3 on OpenAI Gym Mujoco environments.
This script follows the settings of http://arxiv.org/abs/1802.09477 as much
as possible.
"""
import statistics
import os
import functools
import argparse
import logging
import sys

import gym
import gym.wrappers
import numpy as np
import pandas as pd 
import torch
from torch import nn
import json


import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
import random 
import logging
import time 

import csv

from TD3_MAS import *

from torch.autograd import Variable

from White_Attacker_TD3 import FGSM, PGD

start = time.time()

def main():

    parser = argparse.ArgumentParser()
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
        "--env",
        type=str,
        default="Hopper-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )

    parser.add_argument('--budget', type=float, default=1)
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--start_atk', type=int, default='1')
    parser.add_argument('--clip', type=bool, default=True,help='If set to False, actions will be projected based on unclipped nominal action ')
    parser.add_argument("--rollout",default="MAS", type =str,choices = [ "MAS", 'BlackBox',"FGSM", 'PGD'], help=" Attack ")
    parser.add_argument("--s",default="l1", type =str,choices = [ "l1", 'l2'], help=" l1 or l2 norm ")




    parser.add_argument("--attack_steps",default=40, type =int, help=" number of steps use to update the attacker ")



    parser.add_argument("--attack",default="range", type =str,choices = [ "range", "fix",'_'], help="attack method ")

    parser.add_argument("--space",  default= "caction",choices = ["caction", "aspace" ,'_'], help=" caction: percent of current action, aspace: percent of action space")

    parser.add_argument("--percentage", type=int, default= 25 , help="attack pecentage ") #choices = [25,50,100, 200,0],

    parser.add_argument("--objective", default= 'action',choices = ["action","obs", "None",'_'], help="attack target")

    parser.add_argument("--direction", default= 'same', choices = ["same", "flip", 'random', 'random_individually','_'], help="attack direction")



    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    #print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)



    def sample_adv_action():
        adv_action = np.random.uniform(action_space.low, action_space.high, action_size)
        adv_action = Variable(torch.from_numpy(adv_action.astype(np.float32)))
        return adv_action
    
    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)

        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    policy = nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        nn.Tanh(),
        pfrl.policies.DeterministicHead(),
    )
    policy_optimizer = torch.optim.Adam(policy.parameters())

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        q_func_optimizer = torch.optim.Adam(q_func.parameters())
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high
    )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.TD3(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )

    spy = TD3_Adversary(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )
    
    epsilon = args.epsilon
    lr = args.lr
    budget = args.budget
    update_steps = int(args.attack_steps)
    phi = lambda x: x.astype(np.float32, copy=False)
    if args.load:

        assert not args.load or not args.load_pretrained
        agent.load(args.load)
        env = make_env(True)
        if args.render:
            env.render()
        max_episode_len = env.spec.max_episode_steps
        #agent.eval_model()    #process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
        final_rewards = {}
        Obs_act_Dataframe  =  pd.DataFrame()
        #print("Fix_attack_of_current_action_same_direction_25%")


        with agent.eval_mode():
            for p in range(100):
                obs = env.reset()
                obs = phi(obs)
                obs_origin = copy.deepcopy(obs)
                R = 0
                t = 0
                episode = 0 
                
                step_number = [] 
                obs_value = []
                action_value= []
                rewards_value = []
                while True:
                    # Uncomment to watch the behavior in a GUI window
                    if args.render:
                        env.render()
                    # MAS attack 
                    if args.rollout == 'MAS':
                        if t < args.start_atk:
                            
                            action = spy.act(obs)
                            if args.clip is True:
                                action = np.clip(action, -1, 1)
                                action = list(action)
                        else:
                            # optimization loop here
                            adv_action = sample_adv_action()
                            action, q, adv_q = spy.act_forward(obs, adv_action)
                            if args.clip is True:
                                action = np.clip(action, -1, 1)
        
                            grad_a = compute_grad(q, adv_q, action, adv_action)
                            adv_action_new = adv_action - (lr * grad_a)
        
                            action, q, adv_q_new = spy.act_forward(obs, adv_action_new)
                            counter = 0
        
                            while np.absolute((adv_action - adv_action_new).detach().numpy()).any() > epsilon and counter < 25:
                                # print('Optimizing')
                                grad_a = compute_grad(adv_q, adv_q_new, adv_action, adv_action_new)
                                adv_action = adv_action_new
                                adv_action_new = adv_action - (lr * grad_a)
                                counter += 1
                            delta = adv_action_new.detach().numpy() - action
                            if args.s == 'l2':
                                proj_spatial_delta = l2_spatial_project(delta, budget)
                            elif args.s == 'l1':
                                proj_spatial_delta = l1_spatial_project2(delta, budget)        
                            proj_action = action + proj_spatial_delta
                            action = np.clip(proj_action, -1, 1)

                    # FGSM Attack 
                    elif args.rollout == 'FGSM':
                        FGSM_attack = FGSM(agent, obs, epsilon)
                        obs_actor_attack = FGSM_attack.fgsm_perturbation()
                        action = agent.act(obs_actor_attack)

                    # PGD Attack 
                    elif args.rollout == 'PGD':
                        PGD_attack = PGD(model=agent, obs=obs, num_steps=update_steps, step_size=lr, eps=epsilon, )
                        obs_actor_attack = PGD_attack.PGD_perturbation(args.clip)
                        action = agent.act(obs_actor_attack)

                    # Black box attack series. Rollout defines the black box attack, and the objective is the targetted channel,  
                    # fix or range attack, meaning either based on the designed attack or a random attack. 
                    # The caction presents the current action space, and aspace denotes the entire action space. 
                    # The direction presents these four directions we implemented. 
                                             
                    elif args.rollout == 'BlackBox':
                        if args.objective == 'action':
                                
                            if args.attack == "fix":
                                if args.space == 'caction':
                                    if args.direction == 'same' :           
                                        attack = 1 + args.percentage * 0.01 
                                        action = agent.act(obs)
                                        action = attack * action    
    
                                    if args.direction == 'flip' :
                                        attack = 1 - args.percentage * 0.01 
                                        action = agent.act(obs)
                                        action = attack * action

                                    if args.direction == 'random' :
                                        sign = random.choice([-1, 1])
                                        attack = 1 - sign * args.percentage * 0.01 
                                        action = agent.act(obs)
                                        action = attack * action
    
                                    if args.direction == 'random_individually' :
                                        action =  agent.act(obs)
                                        action = [(1 + args.percentage * 0.01 * random.choice([-1, 1])) * x for x in action]                                   
    
                                if args.space == 'aspace': 
    
                                    if args.direction == 'same' :  
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]     
                                        action = agent.act(obs)    
                                        action_temp = [] 
    
                                        for i in range(len(action)): 
    
                                            if action[i] < 0 :
                                                a = action[i] - attack[i]
                                                action_temp.append(a)
    
                                            elif action[i] > 0 :
    
                                                b = action[i] + attack[i]
                                                action_temp.append(b)
    
                                        action = np.array(action_temp)

    
                                    if args.direction == 'flip' :
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
                                        action = agent.act(obs)    
                                        action_temp = []     
                                        for i in range(len(action)):     
                                            if action[i] < 0 :
                                                a = action[i] + attack[i]
                                                action_temp.append(a)    
                                            elif action[i] > 0 :    
                                                b = action[i] - attack[i]
                                                action_temp.append(b)
    
                                        action = np.array(action_temp)

                                    if args.direction == 'random':
                                        sign = random.choice([-1, 1])
                                        attack = [sign * args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
                                        action = agent.act(obs)
                                        action = attack +  action

                                    if args.direction == 'random_individually' :
                                        attack = [args.percentage * 0.02 for x in range(np.shape(env.action_space)[0])]
                                        new_attack = [ random.choice([-1, 1]) * x for x in attack]
                                        action =  agent.act(obs)    
                                        action =  action + new_attack

                            if args.attack == "range":
                                range_number = args.percentage * 0.01 
                                attack = [random.uniform(-range_number,range_number) for x in range(np.shape(env.action_space)[0])]
                                action = agent.act(obs)
                                action = action + attack
    
                        if args.objective == 'obs':    
                            if args.attack == "fix":    
                                if args.space == 'caction':
                                    if args.direction == 'same' :      
                                        obs = (1+ args.percentage * 0.01 )  * obs 
                                        obs = phi(obs)    

                                    elif args.direction == 'flip' :
                                        obs = (1 - args.percentage * 0.01 )  * obs 
                                        obs = phi(obs)

                                    elif args.direction == 'random' :
                                        sign = random.choice([-1, 1])
                                        obs = (1+ sign * args.percentage * 0.01 ) * obs
                                        obs = phi(obs)

                                    elif args.direction == 'random_individually' :
                                        obs = np.array([(1 + args.percentage * 0.01 * random.choice([-1, 1]))* x for x in obs])
                                        obs = phi(obs)
    
                                elif args.space == 'aspace':     
                                    if args.direction == 'same' : 
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
                                                if args.zeros == 'zero_zero':
                                                    b = obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_zero'
    
                                                elif args.zeros == 'zero_negative':    
                                                    b = obs[i] - attack_obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_negative'
        
                                                elif args.zeros == 'zero_postive':    
                                                    b = obs[i] + attack_obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_postive'
    
    
                                        #print('zeros', list(obs).count(0))
                                        #print('obs', np.shape(obs))
                                        obs = np.array(obs_temp)    
                                        obs = phi(obs)
                                    
                                    elif args.direction == 'flip' : 
                                        attack_obs = [args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])] 
                                        obs_temp = []

                                        for i in range(len(obs)): 
                                            if obs[i] < 0 :
                                                a = obs[i] + attack_obs[i]
                                                obs_temp.append(a)
    
                                            elif obs[i] > 0 :
                                                b = obs[i] - attack_obs[i]
                                                obs_temp.append(b)
        
                                            elif obs[i] == 0 :
                                                b = obs[i]    
                                                if args.zeros == 'zero_zero':
                                                    b = obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_zero'
    
                                                elif args.zeros == 'zero_negative':
                                                    b = obs[i] + attack_obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_negative'
    
                                                elif args.zeros == 'zero_postive':    
                                                    b = obs[i] - attack_obs[i]
                                                    obs_temp.append(b)
                                                    name = 'zero_postive'

                                        # print("len(obs)", len(obs))
                                        obs = np.array(obs_temp)
                                        obs = phi(obs)
                                        #print("len(obs)", len(obs))
                                        #print("count(obs)", list(obs).count(0))

                                    elif args.direction == 'random' :
                                        sign = random.choice([-1, 1])
                                        attack_obs = [sign * args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])]  
                                        obs = obs + attack_obs       
                                        obs = phi(obs)

                                    elif args.direction == 'random_individually' :
                                        #print('obs11', obs)
                                        attack_obs = [ args.percentage * 0.02 for x in range(np.shape(env.observation_space)[0])]
                                        new_attack_obs = [ random.choice([-1, 1]) * x for x in attack_obs]
                                        obs = obs + new_attack_obs  
                                        obs= phi(obs)
    
                            if args.attack == "range":
                                range_number = args.percentage * 0.01 
                                attack_obs = [random.uniform(-range_number,range_number) for x in range(np.shape(env.observation_space)[0])]
                                obs = obs + attack_obs
                                obs = phi(obs)

                            action = agent.act(obs)
                            zero_number_obs = (len(action),list(action).count(-1))

                    if args.objective == 'None':
                        action = agent.act(obs)
                    obs_value.append(obs)
                    action_value.append(action)
                    step_number.append(t)
                    obs, r, done, _ = env.step(action)
                    obs = phi(obs)
                    R += r
                    rewards_value.append(r)
                    t += 1
                    reset = t == max_episode_len                    
                    #print("done", done)
                    #print("reset", reset)
                    agent.observe(obs, r, done, reset)

                    if done or reset:
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

        env = make_env(test=False)

        obs = env.reset()
        phi = lambda x: x.astype(np.float32, copy=False)
        obs = phi(obs)
        values_episodes = {}
        episodes_number = []

        while t <= args.steps:
            #if args.render:
                #env.render()

            attack = [random.uniform(-1,1) for x in range(np.shape(env.action_space)[0])]
            print( attack)
            action = agent.act(obs)
            # print("action", action)
            # print("action type", np.shape(action))
            # print("action steps", t)
            obs, r, done, info = env.step(action)
            obs = phi(obs)
            t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len  == max_episode_len
            agent.observe(obs, r, done, reset)

            if done or reset:
                #print('------------episode:', episode_idx, 'R:', episode_r)
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


        final_rewards ={}
        for i in range(len(episodes_number)):
            temp = []
            for y in range(1,11):
                temp.append(values_episodes[str(episodes_number[i]-y)])

            final_rewards[str(episodes_number[i])] = temp
            episodes_number[i]
       
        # print('episodes_number.',episodes_number)
        # print('values_episodes.',values_episodes)
        # print('final_rewards.',final_rewards)

        save_dir =  'withoutattack' + args.outdir + '/'
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
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            adr = os. getcwd()+ '/'       
            store_data( df = Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli' )
        
        elif args.rollout == 'MAS':

            save_folder =  'MAS_Attack' + '/' + "Starting_Point_" + str(args.start_atk) + '_' + "Clip_"+str(args.clip ) + '_Learning_rate_' +str(args.lr ) +\
                '_epsilon_' + str(args.epsilon)+ '_budget_' + str(args.budget)+ '_norm_' + str(args.s)+'/'+ str(args.env)[:-3] +'/'              
            save_result =  'Seed' + str(args.seed)+ '.parquet'

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            adr = os. getcwd()+ '/'
            store_data( df = Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli' )
        

        elif args.rollout == 'FGSM':
            save_folder =  'FGSM_Attack' + '/' + 'Epsilon_' + str(args.epsilon) +'/'+ str(args.env)[:-3] +'/'                
            save_result =  'Seed' + str(args.seed)+ '.parquet'            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            adr = os. getcwd()+ '/'                
            store_data( df = Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli')

        elif args.rollout == 'PGD':
            save_folder =  'PGD_Attack' + '/' + 'Epsilon_' + str(args.epsilon)+'_Learning_rate_' +str(args.lr ) + "_Update_steps_" +str(args.attack_steps ) +'/'+ str(args.env)[:-3] +'/'        
            save_result =  'Seed' + str(args.seed)+ '.parquet'            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            adr = os. getcwd()+ '/'
            store_data( df = Obs_act_Dataframe, fn=adr + os.path.join(save_folder,save_result ), compression='brotli')




def store_data(df, fn, compression=''):
    print('writing...', end='', flush=True)
    compression = 'brotli' if compression == '' else compression
    df.to_parquet(fn, engine='pyarrow', compression=compression)

def compute_grad(q, adv_q, action, adv_action):
    if type(action) == torch.Tensor:
        action = Variable(action)
    else:
        #print(type(action))
        action = Variable(torch.from_numpy(action))
    diff_q = (q - adv_q)
    diff_a = action - adv_action
    diff_a = diff_a.data[0]
    diff_a[diff_a == 0] = 1e-3
    diff_a = Variable(diff_a)
    grad = diff_q / diff_a
    return grad

    
if __name__ == "__main__":
    main()
    end = time.time()
    print('Total time cost{}'.format(end- start))