# RL Natural Robustness
Repository for implementing adversarial attack for reinfrocement learning 

## Abstract
Deep reinforcement learning (DRL) has been shown to have numerous potential applications in the real world. However, DRL algorithms are still extremely sensitive to noise and adversarial perturbations, hence inhibiting the deployment of RL in many real-life applications. Analyzing the robustness of DRL algorithms to adversarial attacks is an important prerequisite to enabling the widespread adoption of DRL algorithms. Common perturbations on DRL frameworks during test time include perturbations to the observation and the action channel. Compared with observation channel attacks, action channel attacks are less studied; hence, few comparisons exist that compare the effectiveness of these attacks in DRL literature. In this work, we examined the effectiveness of these two paradigms of attacks on common DRL algorithms and studied the natural robustness of DRL algorithms towards various adversarial attacks in hopes of gaining insights into the individual response of each type of algorithm under different attack conditions. 


## Experiments Table 

<center>
<p align="center">
    <img src="/images/Stack bar plots/Blackbox table.png" width="800">
</p>

The figure above shows the various black-box strategies implemented. The attacks can be mounted on either one of the two channels, with the constraint on the attack following one of the three magnitudes and the specific instantiation following one of the four directions. 


We implemented the three white-box attack strategies tested on the common benchmark RL algorithms. The white-box attacks we implemented are the Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) algorithm and the Myopic Action Space (MAS) attack algorithm.


## Results  

<p align="center">
    <img src="/images/tend plots/BlackBox line plot dotted.png" width="800">
</p>

The figures above shows the results of black box attack. The x-axis represents $\epsilon$ and the y-axis represents the $\% \Delta R$ with respect to $\epsilon$. The solid line represents the average $\% \Delta R$ across all black box attacks and environments, and the dotted line represents the average excluding the Ant environment. The figure in left shows the action channel attack, and the figure in the right shows the observation channel attack.

<p align="center">
    <img src="/images/FGSM Attack Line Plot/agent variation along .png" width="600">
    <img src="/images/PGD Attack Line Plot/agent variation along .png" width="600">
    <img src="/images/MAS Attack Line Plot Epsilon/agent variation along Epsilon 0.3.png" width="600">
</p>

The figures above shows the results of white box attack. The plots show the relationship between the value of $\epsilon$ (x-axis) and $\% \Delta R$ (y-axis). Line markers in the plots represent experiments we ran with a specific value of $\epsilon$. The solid line represents the average $\% \Delta R$ across all environments, and the dotted line represents the average excluding the Ant environment, and FGSM, PGD, and MAS attacks are shown from top to bottom.
</center>



## Summary 

<p align="center">
    <img src="/images/Bubble chart/Bubble_chart.png" width="650">
</p>

The figure above summarizes the algorithms' correlation between sensitivity and robustness.



## Install
clone repo and install MuJoCo and PyTorch in a Python>=3.6.13 environment.
~~~
git clone https://github.com/super864/Natural-Robustness-RL.git # clone
~~~
install MuJoCo and install PyTorch 1.10.2
refer https://github.com/openai/mujoco-py and https://pytorch.org/


## Running black box attack 

Example run for running black box attack series: 

under ddpg directory
~~~
python black_box_attack.py --agent ddpg
~~~

Each Algorithm directory has its own black_box_attack.py file to run 


## Running white box attack 

Example run for running white box attack series: 

under ddpg directory
~~~
python Whitebox_DDPG_runner.py --agent ddpg
~~~

Each Algorithm directory has its own whitebox runner file to run 


## Running individual attack 

under ppo directory

Example run for black box attack 
~~~
python train_ppo.py --load /mnt/raid10/Natural-Robustness-RL/ppo/withoutattackHalfCheetah/attack_HalfCheetah_seed_0/20210709T093820.456842/HalfCheetah-v2_0/agent  --env HalfCheetah-v2 --gpu -1 --seed 0 --objective action --attack fix --space caction --direction same --percentage 5 --rollout BlackBox
~~~

Example run for white box attack 
~~~
python train_ppo.py --load /mnt/raid10/Natural-Robustness-RL/ppo/withoutattackHalfCheetah/attack_HalfCheetah_seed_0/20210709T093820.456842/HalfCheetah-v2_0/agent  --env HalfCheetah-v2 --gpu -1 --seed 0 --rollout PGD --epsilon 0.1 --clip True --lr 3 --attack_steps 25
~~~
