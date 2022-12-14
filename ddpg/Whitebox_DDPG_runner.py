import os 
import sys
import subprocess
import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    type=str,
    default=r"",
    help=(
        "Algorithm to test."
    ),
)
args = parser.parse_args()

start = time.time()


# finding the path for the corresponding policy
def Agent_Path(path):
    a = []
    for root, dir, files in os.walk(path):
        for name in dir:
            if name == "agent": 
                a.append(os.path.join(root, name))
            else: 
                Agent_Path(os.path.join(root, name))
    return a

# sorted the path
def sort_function(a):
    a.sort()

# generate the corresponding running command line 
def test(address,seed,rollout,clip,start_atk,epsilon,lr,budget, norm, steps_number= None):
    test_run = []
    for key, values in address.items():

        for v in range(len(values)):

            for j in rollout:
                for l in epsilon:
                    if j == "FGSM":

                        run_command = "python train_" + args.agent +".py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --epsilon {4}".format(values[v],key+"-v2",v,j, l)
                                        
                        test_run.append(run_command)
                        continue
                    for u in clip:
                        for d in lr:
                            if j == "PGD":
                                for stp in steps_number: 

                                    run_command = "python train_" + args.agent +".py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --epsilon {4} --clip {5} --lr {6} --attack_steps {7}".format(values[v],key+"-v2",v,j, l,u, d, stp)

                                    test_run.append(run_command)

                                continue


                            for k in start_atk:
                                    for ze in budget:
                                        for s in norm:
                                            run_command = "python train_" + args.agent +".py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --clip {4} --start_atk {5} --epsilon {6} --lr {7} --budget {8} --s {9}".format(values[v],key+"-v2",v,j,u,k,l,d,ze,s)
                                        
                                            test_run.append(run_command)

    return test_run


# generating dictionary with key environment and value of the directory
def address(Directory_Path,env_list,env_name):
    temp= {}
    for i in range(len(env_list)): 

        Directory_Path_indivial = Directory_Path   + env_list[i] + "/"
        print("Directory_Path_indivial", Directory_Path_indivial)
        #print("env_list", env_list)
        a = Agent_Path(Directory_Path_indivial)
        a.sort()
        temp[env_name[i]] = a
    return temp


def main():
    env_list = ['withoutattackHalfCheetah', 'withoutattackAnt', 'withoutattackHopper', 'withoutattackSwimmer', 'withoutattackWalker']

    env_name =['HalfCheetah', 'Ant', 'Hopper', 'Swimmer', 'Walker2d']

    seed = list(range(6))

    rollout= ["PGD", "FGSM", "MAS"]
    
    clip=[True]
    
    start_atk = [1]

    epsilon = [0.1,0.05,0.15,0.2,0.25,0.3,0.35,0.4]

    lr = [3]

    budget = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    norm =["l1"] 

    steps_number = [25]

    directory_path = os.getcwd() +'/'

    all_agent_address = address(directory_path,env_list,env_name) 

    test_run = test(all_agent_address,seed,rollout,clip,start_atk,epsilon,lr,budget,norm,steps_number)


    for i in test_run:
        print(i )
        subprocess.call( i, shell = True)


if __name__ == "__main__":
    main()
    end = time.time()
    print('Total time cost{}'.format(end- start))