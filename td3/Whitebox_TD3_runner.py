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


def test(address,seed,rollout,clip,start_atk,epsilon,lr,budget,norm,steps_number):

    test_run = []
    for key, values in address.items():
        
        for v in range(len(values)):

            for j in rollout:
                for l in epsilon:
                    if j == "FGSM":

                        run_command = "python train_td3.py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --epsilon {4}".format(values[v],key+"-v2",v,j, l)
                                        
                        test_run.append(run_command)
                        continue
                    for u in clip:
                        for d in lr:
                            if j == "PGD":
                                for stp in steps_number: 

                                    run_command = "python train_td3.py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --epsilon {4} --clip {5} --lr {6} --attack_steps {7}".format(values[v],key+"-v2",v,j, l,u, d, stp)

                                    test_run.append(run_command)

                                continue


                            for k in start_atk:
                                    for ze in budget:
                                        for s in norm:
                                            run_command = "python train_td3.py --load {0}  --env {1} --gpu -1 --seed {2} --rollout {3} --clip {4} --start_atk {5} --epsilon {6} --lr {7} --budget {8} --s {9}".format(values[v],key+"-v2",v,j,u,k,l,d,ze,s)
                                        
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

    env_name =['HalfCheetah', 'Ant', 'Hopper', 'Swimmer', 'Walker2d']

    seed = list(range(6))

    rollout= ["PGD"] #["FGSM"]#["MAS"]#
    
    clip=[True] #[True, False]
    
    start_atk = [1] #[1,2,3] #

    epsilon = [0.1,0.05,0.15,0.2,0.25,0.3,0.35,0.4]#[0.1,0.2,0.4,0.8] #[0.1,0.2,0.3]#

    lr = [3]# [3,4,5,6]

    budget = [1,2,4,8]

    
    directory_path = r"/home/scslab/Desktop/pfrl/td3/"

    all_agent_address = address(directory_path,env_list,env_name) 

    norm =["l1"] # ["l1", "l2"]

    if "PGD" in rollout:
        steps_number = [25]
    else:
        steps_number= None
    #print('all_agent_address', all_agent_address)


    test_run = test(all_agent_address,seed,rollout,clip,start_atk,epsilon,lr,budget,norm,steps_number)




    for i in test_run:
        print(i )

        subprocess.call( i, shell = True)


    #print("test_run", "\n",test_run[:30#[1],"\n", test_run[2],"\n",test_run[3],"\n",test_run[4])


    # for i in test_run:
    #     print(test_run[:10])


    # returnStr = ''
    # for item in test_run:
    #     returnStr += str(item)+' '
    # print(returnStr)




    # #!/usr/bin/env python
    # import sys
    # saw_errors = 0
    # for k, v in temp.items():
    #     if '\0' in k or '\0' in v:
    #         saw_errors = 1 # setting exit status is nice-to-have but not essential
    #         continue       # ...but skipping invalid content is important; otherwise,
    #                        #    we'd corrupt the output stream.
    #     sys.stdout.write('%s\0%s\0' % (k, v))
    # sys.exit(saw_errors)



# for key in temp :
#     print( key , "->", temp[key] )
#     print( '\n')






if __name__ == "__main__":
    main()
    end = time.time()
    print('Total time cost{}'.format(end- start))
