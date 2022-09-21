# -*- coding: utf-8 -*-
import pandas as pd 
#%%

import numpy as np 
import matplotlib.pyplot as plt 
import statistics
import seaborn as sns 
import os
import argparse
import functools
import math

# from pylab import plot, show, savefig, xlim, figure, \
#                 hold, ylim, legend, boxplot, setp, axes

from ast import literal_eval
from copy import deepcopy
import pylab
import matplotlib.patches as mpatches
from matplotlib import pyplot
from matplotlib.colors import ListedColormap

from matplotlib.patches import PathPatch
from matplotlib.patches import Patch

from collections import defaultdict

#%%
# walk through all the folder to return the file directories and folder names 

def walk_files(path,Environment_filter=None):  
    file_path_1= []
    folder_name_1 = []
    #Environment_filter  = ["Ant","HalfCheetah", "Hopper", "Swimmer", "Walker2d"] #= ["zero_negative" ,"zero_positive" ]
    
    for root,dirs,files in os.walk(path):
        for file in files:
            file_path = os.path.join(root,file)
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_name = os.path.basename(file_path)
            check = set(folder_name_1)
            isMatch = [ True for x in Environment_filter if x in file_path]
             
            if folder_name in check : 
                Itemnumber = folder_name_1.index(folder_name)

            else: 
                if True in isMatch:
                    Itemnumber = len(folder_name_1)
                    folder_name_1.append(folder_name)
                    
            if True in isMatch:
                
                file_path_1.append(file_path)

    
    file_path_1.sort()
    folder_name_1.sort()

    return file_path_1,folder_name_1


#%%
# group all valid path int to one dictionary, input: file directories and folder names
# filter_condition is the filter out unwanted case
def regroup_address(file_path, file_name,filter_condition = None, zero_condition = None ):
    group_file = {}
    
    if filter_condition is not None:
   
        for i in list(file_path): 
            
            isMatch = [ True for x in filter_condition if x in i]
    
            if True in isMatch:
                file_path.remove(i)
                
                
    if zero_condition is not None:
   
        for i in list(file_path): 
            
            isMatch = [ True for x in zero_condition if x not in i or "Zero_Pixels"  in i]
    
            if isMatch.count(True) == 3:
                file_path.remove(i)
            

            
    for i in file_name: 
        group_file[i] = np.array([])
    for i in file_path:        
        for y in file_name:            
            if y in i:
                group_file[y] = np.append(group_file[y],i)
                    
    return group_file

#%%
# 

def All_seeds_rewards(file_all, Environments, test = True, zero_condition = False ):
    #labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]

    files_results = [] 
    files_name = []
    
    if zero_condition: 
        
        file = defaultdict(list)
        for key,file_path in file_all.items():
            percent = ["25%", "50%", "100%", "200%"]
            
            

            for per in percent: 
                if per in key:
                    for p in file_path:
                        file[key].append(p)


        
        for key,file_path in file.items():
            
            
            over_file_rewards = {}
            agents  = ["Ant-v2zero_negative","Ant-v2zero_positive", "zero_zero"] #= ["Ant-v2zero_negative","Ant-v2zero_positive", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]

            for i in agents: 
                over_file_rewards[i] = np.array([]) 
                
            
            for name in file_path:        

                for i in agents:
                    if i in name and "Zero_Pixels" not in name : 
                        tem_name = i
                
                        RL_data = read_file(name)
                                
                        columns = RL_data.columns.tolist()
                        print("tem_name",tem_name)
            
                        #print("RL_data",RL_data)  
                        #print("columns",columns)

                        
                        if test:
                            columns_values = RL_data.iloc[0].values.tolist()
                            #print("over_file_rewards[tem_name]",over_file_rewards[tem_name])
                            #print("columns_values",columns_values)
                            over_file_rewards[tem_name]= np.append(over_file_rewards[tem_name],columns_values)
            
            
                        else: 
                            columns_values = RL_data.iloc[0].apply(literal_eval).values
                            average_rewards = []
                            for i in columns_values:
                               rewards_avg =  Average_rewards(i)
                               average_rewards.append(rewards_avg)
                               #print(rewards_avg)
                        
                            over_file_rewards.append(average_rewards)


            #print('over_file_rewards',over_file_rewards)

            files_results.append(over_file_rewards)
            files_name.append(key)

        
    else:
        for key,file_path in file_all.items():
    
            over_file_rewards = {}
            
            for i in Environments: 
                over_file_rewards[i] = np.array([]) 
                
                
    # =============================================================================
    #         seed_address = defaultdict(list)
    #         for  i in file_path:
    #             
    #             for env in labels_environment:
    #                 if env in i :
    #                     
    #                     seed_address[env].append(i)
    #                     
    # 
    #                     
    #         seed = [] 
    # 
    #         for env,address in seed_address.items(): 
    #             
    #             seed_value = []
    #             
    #             for i in address: 
    # 
    #                 RL_data = read_file(i)
    #                 mean_values = RL_data.loc[0].values
    #                 
    #                 seed_value.append(mean_values)
    #             
    #             
    #             mean_all = []
    #             std_all = []
    #             for i in range(100):
    #                 g = []
    # 
    # 
    #                 for y in range(6):
    #                     a = seed[y][i]
    #                     g.append(a)
    # 
    # 
    #                 #print("gg",g)
    #                 mm= statistics.mean(g)
    #                 std = statistics.stdev(g)
    #                 mean_all.append(mm)
    #                 std_all.append(std)
    #         
    #         
    # =============================================================================
                 
                
            
            for name in file_path:        
                
                RL_data = read_file(name)
                        
                for i in Environments:
                    if i in name : 
                        tem_name = i
                
                if test:
                    columns_values = RL_data.iloc[0].values.tolist()
                    over_file_rewards[tem_name]= np.append(over_file_rewards[tem_name],columns_values)
    
    
                else: 
                    columns_values = RL_data.iloc[0].apply(literal_eval).values
                    average_rewards = []
                    for i in columns_values:
                       rewards_avg =  Average_rewards(i)
                       average_rewards.append(rewards_avg)
                       #print(rewards_avg)
                    over_file_rewards[tem_name]= np.append(over_file_rewards[tem_name],average_rewards)
    
    
            files_results.append(over_file_rewards)
            files_name.append(key)

        
    return files_results, files_name


#%%
def Renaming(graph_name):
        
    temp_name_1 = ["action_fix_attack_aspace","action_fix_attack_caction", "obs_fix_attack_aspace", "obs_fix_attack_caction","action_range_attack___", "obs_range_attack___" ]
    temp_name = ["same", "flip", "random"]
    

    
    for idx,i in enumerate(graph_name): 

        for hh in temp_name_1:
            if i.startswith(hh): 
                for y in temp_name:
                    if y in i :
                        #print(i)
                        if "action_" in i and "caction" in i:
                            graph_name[idx]=i.replace(hh +y, "action attack respect to current action space with "+y+" direction ")
                        elif "action_" in i and "aspace" in i:   
                            graph_name[idx]=i.replace(hh + y , "action attack respect to action space with "+y+" direction ")
       
                        
                        elif "obs_" in i and "caction" in i:
                            graph_name[idx]=i.replace(hh +y, "observation attack respect to current observation space with "+y+" direction ")
                        elif "obs_" in i and "aspace" in i:   
                            graph_name[idx]=i.replace(hh + y , "observation attack respect to observation space with "+y+" direction ")
                        elif "obs_" in i and "range_" in i:   
                            graph_name[idx]=i.replace(hh + y , "observation attack respect to observation space with "+y+" direction ")
                    
                    if "action_" in i and "range_" in i:   
                        graph_name[idx]=i.replace(hh , "action range attack ")
                    
                    elif "obs_" in i and "range_" in i:   
                        graph_name[idx]=i.replace(hh , "observation range attack ")
                        
    for idx,i in enumerate(graph_name):  
                   
        if "_individually" in i :
            print(i)
            graph_name[idx] = i.replace("_individually", "individually ")

    return graph_name
                        
                        
                        

                
                        

#%%



def None_attack(PPO_files_results, PPO_files_name, DDPG_files_results,DDPG_files_name,SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name):
    
    
    
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    PPO_None_dic = {}
    DDPG_None_dic = {}
    SAC_None_dic= {}
    TD3_None_dic= {}
    TRPO_None_dic= {}
 
    for nn, rr in zip(PPO_files_results,PPO_files_name):
        
        if "None" in rr:
            DDPG_None = List_indx(DDPG_files_name, rr)
            SAC_None = List_indx(SAC_files_name, rr)
            TD3_None = List_indx(TD3_files_name, rr)
            TRPO_None = List_indx(TRPO_files_name, rr)

            for elements in labels_environment:                
                                
                DDPG_None_data = DDPG_files_results[DDPG_None[0]][elements]
                SAC_None_data = SAC_files_results[SAC_None[0]][elements]
                TD3_None_data = TD3_files_results[TD3_None[0]][elements]
                TRPO_None_data = TRPO_files_results[TRPO_None[0]][elements]
                
                PPO_None_dic[elements] = nn[elements]
                DDPG_None_dic[elements] = DDPG_None_data
                SAC_None_dic[elements] = SAC_None_data
                TD3_None_dic[elements] = TD3_None_data
                TRPO_None_dic[elements] = TRPO_None_data
    
                
    return PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic



def data_concatenation(PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic, PPO_files_results, PPO_files_name, DDPG_files_results,DDPG_files_name,SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name, graph_name, method= "percentage differeces"):
    data_frame = pd.DataFrame()
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    


    for ppo,yy,plot_name in zip(PPO_files_results,PPO_files_name, graph_name):
        if "None" not in yy:
        
        
            
            DDPG_list = List_indx(DDPG_files_name, yy)
            SAC_list = List_indx(SAC_files_name, yy)
            TD3_list = List_indx(TD3_files_name, yy)
            TRPO_list = List_indx(TRPO_files_name, yy)
            
            
            labels_agent = ["PPO", "DDPG", "TRPO", "TD3", "SAC"]
            
            
            if method != "zero":
                for ID, elements in enumerate(labels_environment):
    
                    if method == "percentage differeces":
                        box_1, box_2, box_3, box_4, box_5 = percentage_differeces(ppo[elements],PPO_None_dic[elements]), percentage_differeces(DDPG_files_results[DDPG_list[0]][elements],DDPG_None_dic[elements]), percentage_differeces(TRPO_files_results[TRPO_list[0]][elements],TRPO_None_dic[elements]), percentage_differeces(TD3_files_results[TD3_list[0]][elements],TD3_None_dic[elements]),percentage_differeces( SAC_files_results[SAC_list[0]][elements],SAC_None_dic[elements])
                        
                    elif method == "average rewards":
                        box_1, box_2, box_3, box_4, box_5 = Average_rewards(ppo[elements]), Average_rewards(DDPG_files_results[DDPG_list[0]][elements]), Average_rewards(TRPO_files_results[TRPO_list[0]][elements]), Average_rewards(TD3_files_results[TD3_list[0]][elements]),Average_rewards( SAC_files_results[SAC_list[0]][elements])
                    
                    
                    data=[box_1, box_2, box_3, box_4, box_5]
    
                    data_frame_temp = pd.DataFrame(data =[data + [elements]],columns=labels_agent+["environment"] )
                    data_frame = data_frame.append(data_frame_temp)
                
                
            elif method == "zero":

                labels_environment = ['zero_zero', 'Ant-v2zero_positive', 'Ant-v2zero_negative']
                for ID, elements in enumerate(labels_environment):
                    
                    
                    box_1, box_2, box_3, box_4, box_5 = ppo[elements], DDPG_files_results[DDPG_list[0]][elements], TRPO_files_results[TRPO_list[0]][elements], TD3_files_results[TD3_list[0]][elements], SAC_files_results[SAC_list[0]][elements]
    
                    box_1_name = np.repeat("PPO", len(box_1))
                    box_2_name = np.repeat("DDPG", len(box_2))
                    box_3_name = np.repeat("TRPO", len(box_3))
                    box_4_name = np.repeat("TD3", len(box_4))
                    box_5_name = np.repeat("SAC", len(box_5))
                    if elements == "zero_zero": 
                        elements_name = "Zero bias noise"
                    
                    if elements == "Ant-v2zero_negative": 
                        elements_name = "Negative bias noise"
                        
                    if elements == "Ant-v2zero_positive": 
                        elements_name = "Positive bias noise"
                        
                    data_frame_ppo = pd.DataFrame({"data" : box_1,"agent":box_1_name,"attack":str(plot_name),"Environment":str(elements_name) } )
                    data_frame_DDPG = pd.DataFrame({"data" : box_2,"agent":box_2_name,"attack":str(plot_name),"Environment":str(elements_name) } )
                    data_frame_TRPO = pd.DataFrame({"data" : box_3,"agent":box_3_name,"attack":str(plot_name),"Environment":str(elements_name) } )
                    data_frame_TD3 = pd.DataFrame({"data" : box_4,"agent":box_4_name,"attack":str(plot_name),"Environment":str(elements_name) } ) 
                    data_frame_SAC = pd.DataFrame({"data" : box_5,"agent":box_5_name,"attack":str(plot_name),"Environment":str(elements_name) } )

                    data_frame = data_frame.append(data_frame_ppo)
                    data_frame = data_frame.append(data_frame_DDPG)
                    data_frame = data_frame.append(data_frame_TRPO)
                    data_frame = data_frame.append(data_frame_TD3)
                    data_frame = data_frame.append(data_frame_SAC)

    if method != "zero" :
        attack_repeat_name = [x for x in graph_name if "None" not in x for i in range(5) ]
        data_frame['attacks_name'] = attack_repeat_name

    return data_frame



#%%

def Graph_plot(data_frame, save_path):
    
    
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]



    for env in labels_environment: 
        
        Environmen_base_dataframe = data_frame.query('environment == ' + '"'+ str(env)+ '"')
        Environmen_base_dataframe = Environmen_base_dataframe.drop("environment",1)
        Environmen_base_dataframe = Environmen_base_dataframe.set_index(["attacks_name"])
          
        Environmen_base_dataframe = Environmen_base_dataframe.T
        Environmen_base_dataframe = Environmen_base_dataframe.reset_index()
        Environmen_base_dataframe = Environmen_base_dataframe.rename(columns = {"index" : "agent_name"})
        
        
        
        fig, ax = plt.subplots(nrows = 2, ncols= 2, figsize= (16,22.5)) #             
        
        #plt.subplots_adjust(left=0.15,bottom=0.1,top=0.85,right=0.95,hspace=0.5,wspace=0.3)
        plt.suptitle(" Precentages Differences under " + str(env)+ " Environment", fontweight= "bold",fontsize=20)
        fig.tight_layout(pad = 3)
        ax = ax.flatten() 
               
        
        percentage = ["25%","50%","100%","200%"]
        for plot_index,y in enumerate(percentage):
            Attack_25 = []
            Attack_25_yname=[]
            for i in Environmen_base_dataframe.columns:
                if y in i:
                    remove_percentage = i.replace(str(y), '')
                    #print(i)
                    Attack_25.append(i)
                    Attack_25_yname.append(remove_percentage)
                    

            ax_1 = ax[plot_index]        
            Attack_25_frame = Environmen_base_dataframe[Environmen_base_dataframe.columns.intersection(Attack_25+["agent_name"])]   
   
            ax_1.tick_params(axis='y', tickdir='in')

    
            # use log scale for "Ant", "HalfCheetah" environment
            if env in ["Ant", "HalfCheetah"]:
                ax_1.set_xscale('symlog')
                
                
                
            Ant_plot = Attack_25_frame.set_index('agent_name')\
                .T.plot(kind='barh', stacked=True,
                colormap=ListedColormap(sns.color_palette("Paired_r", 5)), 
                figsize=(10,22.5),ax =ax_1, width=0.6,edgecolor="black",alpha=0.5)
    
            
                
            ax_1.set_title("attack for " + str(y) , fontsize=16)
            
            hatch="/"
            for label_range in range(len(ax_1.yaxis.get_ticklabels())):
                if "obs" in ax_1.yaxis.get_ticklabels()[label_range].get_text():

                    for contrainers in Ant_plot.containers:
                        for patch in [contrainers.patches[label_range]]:
                            patch.set_hatch(hatch)
                    

        for number, axis in enumerate(ax):
            
            x_max = max(max(ax[3].get_xlim()),max(ax[2].get_xlim()),max(ax[1].get_xlim()),max(ax[0].get_xlim()))
            x_min = min(min(ax[3].get_xlim()),min(ax[2].get_xlim()),min(ax[1].get_xlim()),min(ax[0].get_xlim()))
            
            plt.setp(axis, xlim= (x_min,x_max)) # have same range x value for all subplots
            #axis.set_xscale('log')
            plt.setp(axis.get_yticklabels(), fontsize=12)
            

            legend = axis.get_legend()

            handles, labels = axis.get_legend_handles_labels()
            handles.append(Patch(facecolor='white', edgecolor='black', hatch=r'///'))
            labels.append("Obervation Attack")
            
            handles.append(Patch(facecolor='white', edgecolor='black'))
            labels.append("Action Attack")
            
            
            axis.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), 
                        loc=2, borderaxespad=0.,edgecolor="black",title= "Algorithm Name")

            
            if number == 0 or number == len(ax)-2 :

                axis.set_yticklabels(Attack_25_yname,fontsize=15)
                axis.set_ylabel('Attack Method', fontsize=15)
            
            elif number%2 !=0:

                axis.get_yaxis().set_visible(False) # no show the second row y-axis 
                
        
            if number != len(ax)-1:

                axis.get_legend().remove()

                
            if number == len(ax)-1 or number == 1:
                axis.set_xlabel( r'$\% \Delta R$', fontsize=15, loc="left", c="black")
                axis.xaxis.set_label_coords(1.015, -0.005)
                

                
                    
        plt.savefig(save_path + "Stack Bar Plot" + str(env) +".png", dpi = 400, bbox_inches = "tight")   

    #adjust_box_widths(fig, 0.9)
    plt.show()
    
    #print('environmen_base_dataframe' ,environmen_base_dataframe )

#%%

def Line_plot(None_attack_combine,labels_environment):
    
    
    
    for env in labels_environment: 
        
        Environmen_base_dataframe = None_attack_combine.query('environment == ' + '"'+ str(env)+ '"')
        Environmen_base_dataframe = Environmen_base_dataframe.drop("environment",1)
        Environmen_base_dataframe = Environmen_base_dataframe.set_index(["attacks_name"])
        Environmen_base_dataframe_name= [x for x in Environmen_base_dataframe.index if "attack 0%" in x]
        Environmen_base_dataframe.drop(Environmen_base_dataframe_name, inplace= True)
        

        
        space = ["action", "observation" ]
        group_name = defaultdict(list)
        
        target = ["range", "respect to current action space", "respect to current observation space","respect to observation space", "respect to action space" ]
        
        direction_range = ["same direction","flip direction","random direction individually","random direction"]
        
        for x in Environmen_base_dataframe.index: 
                                    
            
            for sp in space: 
                if sp in x: 
                    
                    for ta in target: 
                        if ta in x:
                        
                            if ta == "range":
                                
                                group_name[str(sp + " " + ta)].append(x)
                                
                            else:
                      
                                for direc in direction_range:
                                    
                                    if direc in x: 
                                        
                                        group_name[str(sp + " " + ta +" " + direc)].append(x)
                                        
                                        break 

            
            
            
        #fig, ax = generate_subplots(len(group_name.keys()),figsize= (16,22.5))
        fig, ax = plt.subplots(nrows = int(len(group_name)/2), ncols= 2, figsize= (16,22.5))
        plt.suptitle(" Precentages Differences under " + str(env)+ " Environment", fontweight= "bold")
        fig.tight_layout(pad = 3)
        plot_index = 0 
        ax = ax.flatten()
        
        
        Environmen_base_None_attack = Non_attack_frame.query('environment == ' + '"'+ str(env)+ '"')
        Environmen_base_None_attack = Environmen_base_None_attack.drop("environment",1)
        Environmen_base_None_attack.set_index("attacks_name", inplace= True )
        
        for attacks, attack_group in group_name.items():
            axes = ax[plot_index]
            print(plot_index)
            plotting_frame = Environmen_base_dataframe.loc[group_name[str(attacks)]]
            
            plotting_frame = pd.concat([plotting_frame,Environmen_base_None_attack])
            plotting_frame['perentage'] = plotting_frame.index.str.rsplit(' ').str[-1].str[:-1].astype(int)
            plotting_frame = plotting_frame.sort_values(['perentage'])

            plotting_frame_vertical  = plotting_frame.melt("perentage", var_name='Agent',  value_name='Rewards')
            
            plotting_frame = plotting_frame_vertical.pivot(index = "perentage", columns= 'Agent', values = "Rewards") 
            

            g = sns.lineplot(data=plotting_frame, markers= True, mfc ='red', ms='8', dashes =False ,  ax=axes) #x="perentage", y="Rewards", hue='Agent', 
            
            #g = sns.lineplot(x="perentage", y="Rewards", hue='Agent', data=plotting_frame, markers= True, mfc ='red', ms='8', dashes =False ,  ax=axes) #x="perentage", y="Rewards", hue='Agent', 



            axes.set_title(str(attacks) , fontsize=13)

            plot_index +=1
            
            
        for number, axis in enumerate(ax):
            axes = ax[number]
            
            axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


            axes.set_ylim([Environmen_base_dataframe.min().min(),Environmen_base_dataframe.max().max()])

            
            if number%2 == 0 and  number !=0:
                ax[number-1].axes.get_xaxis().set_visible(False)
                #ax[plot_index-1].axes.get_yaxis().set_visible(False)
                
            #plt.close(g.fig)
                
            if number != 1:
                ax[number].get_legend().remove()
            
                
            
            
        plt.show()

        fig.savefig(save_path + "agent variation along with "+ str(env) +".png", dpi = 400, bbox_inches = "tight")    
    
#%%
def summation_plot(data_frame, save_path):
    fig, ax = plt.subplots(nrows = 2, ncols= 2, figsize= (25,16)) #             

    plt.suptitle(" Precentages Differences " + "summation"+ " Environments", fontweight= "bold")
    fig.tight_layout(pad = 3)
    ax = ax.flatten() 
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    data_frame_attackindex = data_frame.set_index('attacks_name')
    
    percentage = ["25%","50%","100%","200%"]
    for plot_index,percent in enumerate(percentage):
        
        ax_1 = ax[plot_index] 
        name = []
        
        for i,z in zip(data_frame['attacks_name'], data_frame['environment']):
            if percent in i and z == "Ant":
                print(i)
                name.append(i)  
    
        agent_oriented_Sum  =  pd.DataFrame()
        for env in labels_environment:
            
            percentage_attack = data_frame_attackindex.loc[name]
            
            percentage_attack_environment = percentage_attack.loc[percentage_attack['environment']==''+env]
            
            percentage_attack_agent_index = percentage_attack_environment.T
            
            percentage_attack_agent_index["sum"] = percentage_attack_agent_index.sum(axis=1)
            
            percentage_attack_agent_index = percentage_attack_agent_index.drop("environment")
            
            percentage_attack_sum = percentage_attack_agent_index.loc[:, "sum"].to_frame()
            
            percentage_attack_sum = percentage_attack_sum.rename({"sum": str(env)}, axis=1)
            
            agent_oriented_Sum = pd.concat([percentage_attack_sum,agent_oriented_Sum], axis =1)
            
        agent_oriented_Sum = agent_oriented_Sum.T  
        agent_oriented_Sum.reset_index(level=0, inplace = True )  
        
        Agent_plot = agent_oriented_Sum.set_index("index")\
            .T.plot(kind='barh', stacked=True,
            colormap=ListedColormap(sns.color_palette("Paired_r", 5)), 
            figsize=(16,22.5),ax =ax_1,edgecolor="black",alpha=0.5)

        show_values_on_bars(Agent_plot,"h", 0.1)
        
        ax_1.set_title("attack for " + str(percent), fontsize=13)  
        
    for number, axis in enumerate(ax):
        show_values_on_bars(axis,"h", 0.1)
        # uncomment for having same range x value for all subplots
        #x_max = max(max(ax[3].get_xlim()),max(ax[2].get_xlim()),max(ax[1].get_xlim()),max(ax[0].get_xlim()))
        #x_min = min(min(ax[3].get_xlim()),min(ax[2].get_xlim()),min(ax[1].get_xlim()),min(ax[0].get_xlim()))
        #plt.setp(axis, xlim= (x_min,x_max))
        
        axis.tick_params(axis='y', labelsize=15)
        axis.tick_params(axis='x', labelsize=13)
        
        
                    
        axis.legend( bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,edgecolor="black",title= "Environment Name")
        
        if number == 0 or number == len(ax)-2 :

            axis.set_ylabel('Algorithm', fontsize=15)
            
        elif number%2 !=0:

            axis.get_yaxis().set_visible(False) # no show the second row y-axis 
            
    
        if number != len(ax)-1:

            axis.get_legend().remove()

            
        if number == len(ax)-1 or number == 1:
            axis.set_xlabel( r'$\% \Delta R$', fontsize=15, loc="left", c="black")
            axis.xaxis.set_label_coords(1.015, -0.005)
    
    plt.savefig(save_path + "agent summation" +".png", dpi = 400, bbox_inches = "tight")              



#%%
def Attack_gap_plot(data_frame, save_path):
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    data_frame_attackindex = data_frame.set_index('attacks_name')
    for env in labels_environment: 
    
        Environmen_base_dataframe = data_frame.query('environment == ' + '"'+ str(env)+ '"')
        Environmen_base_dataframe = Environmen_base_dataframe.drop("environment",1)
        Environmen_base_dataframe = Environmen_base_dataframe.set_index(["attacks_name"])
        
        Environmen_base_dataframe_5  = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("5%")]
        Environmen_base_dataframe_10 = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("10%")]
        Environmen_base_dataframe_15 = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("15%")]
        Environmen_base_dataframe_20 = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("20%")]
        Environmen_base_dataframe_25 = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("25%")]
        Environmen_base_dataframe_50 = Environmen_base_dataframe[Environmen_base_dataframe.index.str.contains("50%")]
        
        Environmen_base_dataframe_5 = pd.concat([Environmen_base_dataframe_5, Environmen_base_dataframe_15,Environmen_base_dataframe_25] )
        Environmen_base_dataframe_5.drop_duplicates(keep=False, inplace=True )
        
        group_percentage = [Environmen_base_dataframe_5,Environmen_base_dataframe_10,Environmen_base_dataframe_15,Environmen_base_dataframe_20,Environmen_base_dataframe_25, Environmen_base_dataframe_50]
  
        differences_10_5 = Environmen_base_dataframe_10.reset_index(drop=True) - Environmen_base_dataframe_5.reset_index(drop=True)
        
        differences_15_10 = Environmen_base_dataframe_15.reset_index(drop=True) - Environmen_base_dataframe_10.reset_index(drop=True)
    
        differences_20_15 = Environmen_base_dataframe_20.reset_index(drop=True) - Environmen_base_dataframe_15.reset_index(drop=True)
    
        differences_25_20 = Environmen_base_dataframe_25.reset_index(drop=True) - Environmen_base_dataframe_20.reset_index(drop=True)
    
        differences_50_25 = Environmen_base_dataframe_50.reset_index(drop=True) - Environmen_base_dataframe_25.reset_index(drop=True)
    
        
        differences_bag= [differences_10_5,differences_15_10,differences_20_15,differences_25_20,differences_50_25]

        yname = []
        for index_name in Environmen_base_dataframe_10.index:
            
            remove_percentage = index_name.replace("10%", '')
            yname.append(remove_percentage)
            
        for d in differences_bag:
            
            d.index = yname            
            
        differences_bag_group = []
        
        for percent_group in differences_bag:
            #print("I")
            percent_group = percent_group.T
            percent_group = percent_group.reset_index()
            percent_group = percent_group.rename(columns = {"index" : "agent_name"})
            differences_bag_group.append(percent_group)
            
        sub_plot_name = ["Differences between 5% and 10% attack","Differences between 10% and 15% attack","Differences between 15% and 20% attack","Differences between 25% and 20% attack","Differences between 25% and 50% attack" ]
        
        fig, ax = plt.subplots(nrows = 2, ncols= 3, figsize= (16,22.5)) #             
        
        #plt.subplots_adjust(left=0.15,bottom=0.1,top=0.85,right=0.95,hspace=0.5,wspace=0.3)
        plt.suptitle(" Precentages Differences under " + str(env)+ " Environment", fontweight= "bold")
        fig.tight_layout(pad = 3)
        ax = ax.flatten()
        plot_index = 0
        for y , name_plots in zip(differences_bag_group, sub_plot_name): 
            #print("plot_index",plot_index)
            ax_1 = ax[plot_index]
            plot_index +=1
        
            Ant_plot = y.set_index('agent_name')\
                .T.plot(kind='barh', stacked=True,
                colormap=ListedColormap(sns.color_palette("Paired_r", 5)), 
                figsize=(16,22.5),ax =ax_1, width=0.6,edgecolor="black",alpha=0.5)
                

            #show_values_on_bars(Ant_plot,"h", 0.1)
            
            ax_1.set_title(str(name_plots), fontsize=13)
            
            hatch="/"
            for label_range in range(len(ax_1.yaxis.get_ticklabels())):
                if "obs" in ax_1.yaxis.get_ticklabels()[label_range].get_text():

                    for contrainers in Ant_plot.containers:
                        for patch in [contrainers.patches[label_range]]:
                            patch.set_hatch(hatch)
        
        for number, axis in enumerate(ax):
            
            x_max = max(max(ax[3].get_xlim()),max(ax[2].get_xlim()),max(ax[1].get_xlim()),max(ax[0].get_xlim()))
            x_min = min(min(ax[3].get_xlim()),min(ax[2].get_xlim()),min(ax[1].get_xlim()),min(ax[0].get_xlim()))
            
            plt.setp(axis, xlim= (x_min,x_max)) # have same range x value for all subplots
            #axis.set_xscale('log')
            plt.setp(axis.get_yticklabels(), fontsize=12)
            
            axis.tick_params(axis='y', labelsize=15)
            axis.tick_params(axis='x', labelsize=13)
            
            legend = axis.get_legend()

            handles, labels = axis.get_legend_handles_labels()
            handles.append(Patch(facecolor='white', edgecolor='black', hatch=r'///'))
            labels.append("Obervation Attack")
            
            handles.append(Patch(facecolor='white', edgecolor='black'))
            labels.append("Action Attack")
            
            
            axis.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), 
                        loc=2, borderaxespad=0.,edgecolor="black",title= "Algorithm Name")
            
            
            
            if number == 0 or number == len(ax)-2 :
    
                axis.set_ylabel('Attack Method', fontsize=15)
            
            if number%3 !=0:
    
                axis.get_yaxis().set_visible(False) # no show the second row y-axis 
                
        
            if number != len(ax)-2:
    
                axis.get_legend().remove()
    
                
            if number == len(ax)-2 or number == 2:
                axis.set_xlabel( r'$\% \Delta R$', fontsize=15, loc="left", c="black")
                axis.xaxis.set_label_coords(1.015, -0.005)
     
                
        fig.delaxes(ax[5])
    
    plt.savefig(save_path + "agent_summation" +".png", dpi = 400, bbox_inches = "tight") 
    
    
    

#%%


def Ant_zero_plot(data_frame, save_path):
    data_frame["percentage"] = data_frame["attack"].str.rsplit(' ').str[-1].str[:-1].astype(int)
    data_frame = data_frame.sort_values(['percentage', 'agent' ])
 
    
    unique_attack = data_frame.attack.unique()
    
    fig, ax = plt.subplots(nrows = int(len(unique_attack)/2), ncols= 2, figsize= (25,15)) #             
    plt.suptitle("Ant Environment Zero Bias Noise", fontweight= "bold" )
    fig.tight_layout(pad = 3)
    ax = ax.flatten() 

    fig.subplots_adjust(hspace=0.3,wspace=0.1)
    
    for idx, u in enumerate(unique_attack):
        data_frame_unique = data_frame.loc[data_frame['attack'] == u]

    
        print(data_frame_unique)
        #exit()
        ax[idx].set_title(str(u), fontsize=13)
        #if idx%2 == 0: 
            
        bplot1 = sns.boxplot(x="agent", y="data", data=data_frame_unique, hue="Environment", ax = ax[idx], showmeans = True, meanline= True, palette='PRGn')
        
        #ax[idx].legend(bbox_to_anchor=(1,0), loc=3, borderaxespad=0, bbox_transform=plt.gcf().transFigure)
        

        
    for number, axis in enumerate(ax):
        
        
        #axis.set_title(fontweight= "bold" )
        sns.boxplot
        axis.set(xlabel=None)
        axis.set(ylabel=None)
        ax[idx].legend( bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,edgecolor="black",title= "Attack condition")
        
        axis.tick_params(axis='y', labelsize=15)
        axis.tick_params(axis='x', labelsize=13)
        
        if number%2 == 0 :

            axis.set_ylabel('Rewards', fontsize=12)
        
        elif number%2 !=0:

            #axis.get_yaxis().set_visible(False) # no show the second row y-axis 
            
            axis.set(xlabel=None)
        if number != len(ax)-1:

            axis.get_legend().remove()

            
        if number == len(ax)-1:
            axis.set_xlabel( 'Algorithm Name', fontsize=12, loc="left", c="black")
            axis.xaxis.set_label_coords(1.015, -0.005)
                
    
    
    plt.savefig(save_path + "Ant bias noise" +".png", dpi = 400, bbox_inches = "tight")
    plt.show()
    plt.close()
    
#%% 
# read csv file 
# input: file directories

def read_file(name):
    
    RL_data = pd.read_csv(name)
    return RL_data

# Average rewards
# input: list of rewards

def Average_rewards(list_rewards):
    return statistics.mean(list_rewards)


# index of a given elemnet 
# input: list, target element 

def List_indx(agent_path, target):
    List_index_return = [agent_path.index(address) for address in agent_path if target in address]   
    return List_index_return


# calculate the percentage difference, first average, then differences  
# input: two lists

def percentage_differeces(agent_data,None_data):
    if len(agent_data) == 0:
        differences = 0
    else:   
        differences = (Average_rewards(agent_data) - Average_rewards(None_data))/ Average_rewards(None_data)
    
    return differences


def concat_dataframe(data_frame,Non_attack_frame): 
    
    attack_combine = pd.concat([data_frame,Non_attack_frame])
        
    return attack_combine
    


def Reorganise_none_attack(None_attack_list,Environments):

    labels_agent = ["PPO", "DDPG", "TRPO", "TD3", "SAC"]
    
    None_attack_dic = defaultdict(list)

    for i, name in zip(None_attack_list, labels_agent):
    
        for elements in Environments:
            
            
                None_attack_dic[str(name)].append(Average_rewards(i[elements]))
     
    Non_attack_frame = pd.DataFrame.from_dict(None_attack_dic)

    Non_attack_frame["environment"] = Environments 
    Non_attack_frame["attacks_name" ]= ["none attack 0%" for x in range(len(Non_attack_frame["environment"]))]
    
    return Non_attack_frame


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() 
                if _x <= 0:
                    _x = p.get_x()  - float(space)* (_x) + p.get_width() 
                    
                else:
                    _x = p.get_x()  + float(space)* (_x) + p.get_width()
                _y = p.get_y() + p.get_height()
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="center",va="bottom")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        

#%%
if __name__ == "__main__":
       

    #os.chdir(r"/home/scslab/Desktop/pfrl/")
    
    #args.PPO 
    PPO =  r'/home/scslab/Desktop/pfrl/ppo/Testing_Result/' 
    
    #args.DDPG 
    DDPG = r'/home/scslab/Desktop/pfrl/ddpg/Testing_Result/' 
    
    #args.SAC 
    SAC =  r"/home/scslab/Desktop/pfrl/sac/Testing_Result/"  
    
    #args.TD3
    TD3 =  r'/home/scslab/Desktop/pfrl/td3/Testing_Result/'

    # args.TRPO
    TRPO = r"/home/scslab/Desktop/pfrl/trpo/Testing_Result/"  
    
    save_path = r"/home/scslab/Desktop/pfrl/ddpg/Different_agent/bar_plots percentage/new/stack bar plot/"
    
#%%   
    # get each algorithim directories and saved name
    
    Environment_filter  = ["Ant","HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    
    PPO_path, PPO_name = walk_files(PPO,Environment_filter)
    
    DDPG_path, DDPG_name = walk_files(DDPG,Environment_filter)
    
    SAC_path, SAC_name = walk_files(SAC,Environment_filter)
     
    TD3_path, TD3_name = walk_files(TD3,Environment_filter)
    
    TRPO_path, TRPO_name = walk_files(TRPO,Environment_filter)

    
#%%    
    same_direction = ["obs_fix_attack_aspacesame25%", "obs_fix_attack_aspacesame50%","obs_fix_attack_aspacesame100%","obs_fix_attack_aspacesame200%"]
      
    flip_direction = ["obs_fix_attack_aspaceflip25%", "obs_fix_attack_aspaceflip50%", "obs_fix_attack_aspaceflip100%", "obs_fix_attack_aspaceflip200","Zero_Pixels"]
 
    #filter_condition = same_direction + flip_direction
    #Environment_filter  = ["Ant","HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    filter_condition = ["Zero_Pixels", 'zero_positive','zero_negative' ] #'zero_zero'
    
    zero_condition = ['zero_positiveted_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4','zero_negative', 'zero_zero']
    
    zero =  False
    
    sum_plot = False 
    
    stack_plot = True
    
    attack_diff_plot = False 
    
    Linear_plot = False
    
    if zero: 
        PPO_addresss = regroup_address(PPO_path,PPO_name,None, zero_condition)
        DDPG_addresss = regroup_address(DDPG_path,DDPG_name,None,zero_condition )
        SAC_addresss = regroup_address(SAC_path,SAC_name,None,zero_condition )
        
                
        TD3_addresss = regroup_address(TD3_path,TD3_name,None,zero_condition )
        TRPO_addresss = regroup_address(TRPO_path,TRPO_name,None,zero_condition )
        
        PPO_files_results, PPO_files_name = All_seeds_rewards(PPO_addresss,Environment_filter,True,zero)
        DDPG_files_results, DDPG_files_name = All_seeds_rewards(DDPG_addresss,Environment_filter,True,zero)
        SAC_files_results, SAC_files_name = All_seeds_rewards(SAC_addresss,Environment_filter,True,zero)
        TD3_files_results, TD3_files_name = All_seeds_rewards(TD3_addresss,Environment_filter,True,zero)
        TRPO_files_results, TRPO_files_name = All_seeds_rewards(TRPO_addresss,Environment_filter,True,zero)
        
        graph_name = deepcopy(PPO_files_name)
        graph_name = Renaming(graph_name)

        PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic  = None_attack(PPO_files_results, PPO_files_name, DDPG_files_results,DDPG_files_name,SAC_files_results, 
                                                                                          SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name)
        
        
        data_frame = data_concatenation(PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic, PPO_files_results, 
                                        PPO_files_name, DDPG_files_results,DDPG_files_name, SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name, graph_name, method ="zero")
        
        Ant_zero_plot(data_frame,save_path)
     
        
    elif sum_plot or stack_plot or attack_diff_plot or Linear_plot : 
        
        PPO_addresss = regroup_address(PPO_path,PPO_name,filter_condition)
        DDPG_addresss = regroup_address(DDPG_path,DDPG_name,filter_condition)
        SAC_addresss = regroup_address(SAC_path,SAC_name,filter_condition)
        TD3_addresss = regroup_address(TD3_path,TD3_name,filter_condition)
        TRPO_addresss = regroup_address(TRPO_path,TRPO_name,filter_condition)    
        
#%%


        PPO_files_results, PPO_files_name = All_seeds_rewards(PPO_addresss,Environment_filter,True)
        DDPG_files_results, DDPG_files_name = All_seeds_rewards(DDPG_addresss,Environment_filter,True)
        SAC_files_results, SAC_files_name = All_seeds_rewards(SAC_addresss,Environment_filter,True)
        TD3_files_results, TD3_files_name = All_seeds_rewards(TD3_addresss,Environment_filter,True)
        TRPO_files_results, TRPO_files_name = All_seeds_rewards(TRPO_addresss,Environment_filter,True)
    
    
        graph_name = deepcopy(PPO_files_name)
        graph_name = Renaming(graph_name)
        
        PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic  = None_attack(PPO_files_results, PPO_files_name, DDPG_files_results,DDPG_files_name,SAC_files_results, 
                                                                                          SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name)
        
        
        data_frame = data_concatenation(PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic, PPO_files_results, 
                                        PPO_files_name, DDPG_files_results,DDPG_files_name, SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name, graph_name)
        
        
        if stack_plot :
            Graph_plot(data_frame, save_path)
        
    
        if sum_plot:
            summation_plot(data_frame,save_path)
            
        if attack_diff_plot:
        
            Attack_gap_plot(data_frame, save_path)
            
            
        if Linear_plot: 

            data_frame = data_concatenation(PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic, PPO_files_results, 
                                            PPO_files_name, DDPG_files_results,DDPG_files_name, SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name, graph_name, method="average rewards" )
        
            None_attack_list = [PPO_None_dic,DDPG_None_dic,SAC_None_dic,TD3_None_dic,TRPO_None_dic]
             
            Non_attack_frame = Reorganise_none_attack(None_attack_list, Environment_filter)
           
            combine_frame = concat_dataframe(data_frame,Non_attack_frame)
            
            
            Line_plot(combine_frame,Environment_filter)
        
    
    