# RL Natural Robustness
Repository for implementing adversarial attack for reinfrocement learning 

<em>**Paper accepted in ???? (???? 2022)**</em>


## Baseline models of experiment results
<p align="center">
    <img src="/images/training_and_testing_images/training/ddpg under Walker2d environment training without attack.png" width="450">
    <img src="/images//training_and_testing_images/testing/ddpg under Walker2d environment testing without attack.png" width="450">
</p>

The figures above shows the Baseline models reslus for (a) training DDPG under Walker environment, (b) Testing DDPG under Walker environment. Since all the other environments perform very similarly regarding the rewards, only one environment is shown here.


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

The figures above shows the results of black box attack. The x-axis represents $\epsilon$ and the y-axis represents the $\% \Delta R$ with respect to $\epsilon$. The solid line represents the average $\% \Delta R$ across all black box attacks and environments, and the dotted line represents the average excluding the Ant environment.

<p align="center">
    <img src="/images/FGSM Attack Line Plot/agent variation along .png" width="600">
    <img src="/images/PGD Attack Line Plot/agent variation along .png" width="600">
    <img src="/images/MAS Attack Line Plot Epsilon/agent variation along Epsilon 0.3.png" width="600">
</p>

The figures above shows the results of white box attack. The plots show the relationship between the value of $\epsilon$ (x-axis) and $\% \Delta R$ (y-axis). Line markers in the plots represent experiments we ran with a specific value of $\epsilon$. The solid line represents the average $\% \Delta R$ across all environments, and the dotted line represents the average excluding the Ant environment.
</center>



## Summary 

<p align="center">
    <img src="/images/Bubble chart/Bubble_chart.png" width="800">
</p>

The figure above summarizes the algorithms' correlation between sensitivity and robustness.



## Install
Clone repo and install requirements.txt in a Python>=3.6.13 enviornment.
~~~
git clone https://github.com/super864/Natural-Robustness-RL.git # clone
~~~
install mujoco and install pytorch 1.10.2
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

Example run for black box attack 
~~~
python train_ppo.py --load /mnt/raid10/Natural-Robustness-RL/ppo/withoutattackHalfCheetah/attack_HalfCheetah_seed_0/20210709T093820.456842/HalfCheetah-v2_0/agent  --env HalfCheetah-v2 --gpu -1 --seed 0 --objective action --attack fix --space caction --direction same --percentage 5 --rollout BlackBox
~~~

Example run for white box attack 
~~~
python train_ppo.py --load /mnt/raid10/Natural-Robustness-RL/ppo/withoutattackHalfCheetah/attack_HalfCheetah_seed_0/20210709T093820.456842/HalfCheetah-v2_0/agent  --env HalfCheetah-v2 --gpu -1 --seed 0 --rollout PGD --epsilon 0.1 --clip True --lr 3 --attack_steps 25
~~~


## Citation
Please cite our paper in your publications if it helps your research:

	@article{article,
	  title={title},
	  author={author},
	  journal={journal},
	  year={2022}
	}

## Paper Links
[Paper Title](paper_link)




## Contributors
- [Qisai Liu](https://github.com/super864/Natural-Robustness-RL)
- [Xian Yeow Lee](https://github.com/xylee95)
