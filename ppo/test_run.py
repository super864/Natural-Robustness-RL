import os 
import sys
import subprocess
import time 

start = time.time()
def Agent_Path(path):

    a = []

    for root, dir, files in os.walk(path):

        for name in dir:

            if name == "agent": 

                a.append(os.path.join(root, name))

            else: 
                Agent_Path(os.path.join(root, name))

    return a

def sort_function(a):
    a.sort()


def test(address,seed,objective,attack,space,direction,percentage,zeros):

    # env=["Walker2d-v2", "Swimmer-v2", "Ant-v2", "HalfCheetah-v2", "Hopper-v2"]

    # out=["Walker" ,"Swimmer", "Ant","HalfCheetah", "Hopper"]

    test_run = []
    for key, values in address.items():

        #print('key', key )

        #print('values', values )
        for v in range(len(values)):


            for j in objective:
                if j == "None":
                    
                    run_command = "python train_ppo_withoutattack.py --load {0}  --env {1} --gpu -1 --seed {2} --objective {3} --attack {4} --space {5} --direction {6} --percentage {7}".format(values[v],key+"-v2",v,j,'_','_','_',0)

                    test_run.append(run_command)
                    continue

                for u in attack:
                    if u == "range":
                        for ii in percentage:

                            run_command = "python train_ppo_withoutattack.py --load {0}  --env {1} --gpu -1 --seed {2} --objective {3} --attack {4} --space {5} --direction {6} --percentage {7}".format(values[v],key+"-v2",v,j,u,'_','_',ii)

                            test_run.append(run_command)
                        continue



                    for k in space:

                        for l in direction:

                            for d in percentage:
                                if k =="aspace" and j == 'obs' and key == "Ant" and l in ['same','flip']:
                                    for ze in zeros:
                                        run_command = "python train_ppo_withoutattack.py --load {0}  --env {1} --gpu -1 --seed {2} --objective {3} --attack {4} --space {5} --direction {6} --percentage {7} --zeros {8}".format(values[v],key+"-v2",v,j,u,k,l,d,ze)
                                    
                                        test_run.append(run_command)
                                    continue

                                run_command = "python train_ppo_withoutattack.py --load {0}  --env {1} --gpu -1 --seed {2} --objective {3} --attack {4} --space {5} --direction {6} --percentage {7}".format(values[v],key+"-v2",v,j,u,k,l,d)

                                test_run.append(run_command)


    return test_run

def address(Directory_Path,env_list,env_name):

    temp= {}

    for i in range(len(env_list)): 

        Directory_Path_indivial = Directory_Path   + env_list[i] + "/"
        print("Directory_Path_indivial", Directory_Path_indivial)
        #print("env_list", env_list)

        a = Agent_Path(Directory_Path_indivial)
        a.sort()
        print("a", a )


        temp[env_name[i]] = a

    return temp




def main():
    env_list = ['withoutattackHalfCheetah', 'withoutattackAnt', 'withoutattackHopper', 'withoutattackSwimmer', 'withoutattackWalker']

    env_name = ['HalfCheetah', 'Ant', 'Hopper', 'Swimmer', 'Walker2d']

    seed = list(range(6))

    objective = ["action", "obs" ,"None"] #

    attack =["range", "fix"] #["fix"] #

    space = ["caction", "aspace"] #["aspace"]

    direction = ['same', 'flip', 'random', 'random_individually']

    percentage = [5,10,15,20,25,50,100,200] #[25,50,100,200] #[5,10,15,20]

    zeros = ["zero_zero", "zero_postive", "zero_negative"]
        
    directory_path = r"/home/scslab/Desktop/pfrl/ppo/"  

    all_agent_address = address(directory_path,env_list,env_name) 



    #print('all_agent_address', all_agent_address)


    test_run = test(all_agent_address,seed,objective,attack,space,direction,percentage,zeros)




    for i in test_run:
        i = i + " --rollout BlackBox"
        print(i )

        subprocess.call( i, shell = True)

if __name__ == "__main__":
    main()
    end = time.time()
    print('Total time cost{}'.format(end- start))
