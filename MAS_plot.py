from pfrl_plots import * 

import pylab
import matplotlib.patches as mpatches
from matplotlib import pyplot
from matplotlib.colors import ListedColormap

from matplotlib.patches import PathPatch
from matplotlib.patches import Patch

from collections import defaultdict

def read_file(name):
    
    RL_data = pd.read_parquet(name, engine='pyarrow')
    
    f = {"rewards" : "sum"}
    RL_data = RL_data[["rewards","episode"]]
    
    g = RL_data.groupby([ 'episode'])
    RL_data = g.agg(f)
    
    
    return RL_data


def data_frame(PPO_files_results, PPO_files_name, DDPG_files_results,DDPG_files_name, SAC_files_results, SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name,None_attack_frame, method="percentage differeces" ):
    data_frame = pd.DataFrame()
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    labels_agent = ["PPO", "DDPG", "TRPO", "TD3", "SAC"]
            
    for ppo,yy in zip(PPO_files_results,PPO_files_name):
        for ID, elements in enumerate(labels_environment):
    
            if method == "percentage differeces":
                box_1, box_2, box_3, box_4, box_5 = percentage_differeces(ppo[elements],None_attack_frame[0][elements]), \
                    percentage_differeces(DDPG_files_results[0][elements],None_attack_frame[1][elements]), \
                        percentage_differeces(TRPO_files_results[0][elements],None_attack_frame[2][elements]),\
                            percentage_differeces(TD3_files_results[0][elements],None_attack_frame[3][elements]),\
                                percentage_differeces( SAC_files_results[0][elements],None_attack_frame[4][elements])
                
            if method == "average rewards":
                box_1, box_2, box_3, box_4, box_5 = Average_rewards(ppo[elements]),\
                    Average_rewards(DDPG_files_results[0][elements]),\
                        Average_rewards(TRPO_files_results[0][elements]), \
                            Average_rewards(TD3_files_results[0][elements]),\
                                Average_rewards( SAC_files_results[0][elements])
            
            
            data=[box_1, box_2, box_3, box_4, box_5]
    
            data_frame_temp = pd.DataFrame(data =[data + [elements]],columns=labels_agent+["environment"] )
            data_frame = data_frame.append(data_frame_temp)
        
    attack_repeat_name = [x for x in PPO_files_name if "None" not in x for i in range(5) ]
    data_frame['attacks_name'] = attack_repeat_name
    return data_frame




def All_seeds_rewards(file_all, Environments, test = True):
    #labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    files_results = [] 
    files_name = []
    
    for key,file_path in file_all.items():

        over_file_rewards = {}
        
        for i in Environments: 
            over_file_rewards[i] = np.array([]) 
            
        
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



def Stack_plots(dataframe,save_path):
    
        
    labels_environment = ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]



    for env in labels_environment: 
        
        Environmen_base_dataframe = dataframe.query('environment == ' + '"'+ str(env)+ '"')
        Environmen_base_dataframe = Environmen_base_dataframe.drop("environment",1)
        Environmen_base_dataframe = Environmen_base_dataframe.set_index(["attacks_name"])
          
        Environmen_base_dataframe = Environmen_base_dataframe.T
        Environmen_base_dataframe = Environmen_base_dataframe.reset_index()
        Environmen_base_dataframe = Environmen_base_dataframe.rename(columns = {"index" : "agent_name"})



        fig, ax = plt.subplots(nrows = 2, ncols= 2, figsize= (16,22.5)) #             
        
        #plt.subplots_adjust(left=0.15,bottom=0.1,top=0.85,right=0.95,hspace=0.5,wspace=0.3)
        plt.suptitle(" MAS Attack under" + str(env)+ " Environment", fontweight= "bold",fontsize=20)
        fig.tight_layout(pad = 3)
        ax = ax.flatten() 
        
        epsilon = [ 0.1, 0.2 , 0.4, 0.8]
        
        
        for plot_index,y in enumerate(epsilon):
            flie_name = []
            plots_name_remove_epsilon=[]
            for i in Environmen_base_dataframe.columns:
                if str(y) in i:
                    remove_epsilon = i.replace("_epsilon_"+ str(y), '')
                    #print(i)
                    flie_name.append(i)
                    plots_name_remove_epsilon.append(remove_epsilon)
                    
                    
            
            ax_current = ax[plot_index]
            Sub_epsilon_frame = Environmen_base_dataframe[Environmen_base_dataframe.columns.intersection(flie_name+["agent_name"])]
            
            Stack_plot = Sub_epsilon_frame.set_index('agent_name').T.plot(kind='barh', stacked=True, \
                    colormap=ListedColormap(sns.color_palette("Paired_r", 5)),\
                        figsize=(10,22.5), ax = ax_current,width=0.6,edgecolor="black",alpha=0.5)
                
                
                
            
            hatch="/"
            for label_range in range(len(ax_current.yaxis.get_ticklabels())):
                if "norm_l2" in ax_current.yaxis.get_ticklabels()[label_range].get_text():

                    for contrainers in Stack_plot.containers:
                        for patch in [contrainers.patches[label_range]]:
                            patch.set_hatch(hatch)
                
                
                
            ax_current.set_title("Epsilon for " + str(y) , fontsize=16)
                
            ax_current.tick_params(axis='y', tickdir='in')
    
            show_values_on_bars(Stack_plot,"h", 00.05)
            
            
        for number, axis in enumerate(ax):
            x_max = max(max(ax[3].get_xlim()),max(ax[2].get_xlim()),max(ax[1].get_xlim()),max(ax[0].get_xlim()))
            x_min = min(min(ax[3].get_xlim()),min(ax[2].get_xlim()),min(ax[1].get_xlim()),min(ax[0].get_xlim()))
            
            plt.setp(axis, xlim= (x_min,x_max)) # have same range x value for all subplots
            #axis.set_xscale('log')
            plt.setp(axis.get_yticklabels(), fontsize=12)
            


            legend = axis.get_legend()
            
            handles, labels = axis.get_legend_handles_labels()
            handles.append(Patch(facecolor='white', edgecolor='black', hatch=r'///'))
            labels.append("L2  Norm")
            
            handles.append(Patch(facecolor='white', edgecolor='black'))
            labels.append("L1 Norm")

            axis.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), fontsize = 12,title_fontsize=15,
                    loc=2, borderaxespad=0.,edgecolor="black",title= "Algorithm Name", prop={'size': 20})
            
            
            
            if number == 0 or number == len(ax)-2 :

                axis.set_yticklabels(plots_name_remove_epsilon,fontsize=15)
                axis.set_ylabel('Attack Method', fontsize=15)
            
            elif number%2 !=0:

                axis.get_yaxis().set_visible(False) # no show the second row y-axis 
                
            if number != len(ax)-1:

                axis.get_legend().remove()

            if number == len(ax)-1 or number == 1:
                axis.set_xlabel( r'$\% \Delta R$', fontsize=15, loc="left", c="black")
                axis.xaxis.set_label_coords(1.015, -0.005)
                
                
                
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + "Stack Bar Plot MAS "+ str(env) +".png", dpi = 400, bbox_inches = "tight")  

    
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
        
if __name__ == "__main__":
       
    #os.chdir(r"/home/scslab/Desktop/pfrl/")
    
    #args.PPO 
    PPO =  r'/home/scslab/Desktop/pfrl/ppo/MAS_Attack/' 
    
    #args.DDPG 
    DDPG = r'/home/scslab/Desktop/pfrl/ddpg/MAS_Attack/' 
    
    #args.SAC 
    SAC =  r"/home/scslab/Desktop/pfrl/sac/MAS_Attack/"  
    
    #args.TD3
    TD3 =  r'/home/scslab/Desktop/pfrl/td3/MAS_Attack/'

    # args.TRPO
    TRPO = r"/home/scslab/Desktop/pfrl/trpo/MAS_Attack/"  
    
    save_path = r"/home/scslab/Desktop/pfrl/MAS Attack plot/"
    
#%%   
    # get each algorithim directories and saved name
    
    Environment_filter  = ["Ant","HalfCheetah", "Hopper", "Swimmer", "Walker2d"]
    
    PPO_path, PPO_name = walk_files(PPO,Environment_filter)
    
    DDPG_path, DDPG_name = walk_files(DDPG,Environment_filter)
    
    SAC_path, SAC_name = walk_files(SAC,Environment_filter)
     
    TD3_path, TD3_name = walk_files(TD3,Environment_filter)
    
    TRPO_path, TRPO_name = walk_files(TRPO,Environment_filter)
    
    
    None_attack_frame = np.load(r'/home/scslab/Desktop/pfrl/Non_attack_frame.npy',allow_pickle=True)
    PPO_addresss = regroup_address(PPO_path,PPO_name,)
    DDPG_addresss = regroup_address(DDPG_path,DDPG_name,)
    SAC_addresss = regroup_address(SAC_path,SAC_name,)
    TD3_addresss = regroup_address(TD3_path,TD3_name,)
    TRPO_addresss = regroup_address(TRPO_path,TRPO_name,)   
    
    
    PPO_files_results, PPO_files_name = All_seeds_rewards(PPO_addresss,Environment_filter,True)
    DDPG_files_results, DDPG_files_name = All_seeds_rewards(DDPG_addresss,Environment_filter,True)
    SAC_files_results, SAC_files_name = All_seeds_rewards(SAC_addresss,Environment_filter,True)
    TD3_files_results, TD3_files_name = All_seeds_rewards(TD3_addresss,Environment_filter,True)
    TRPO_files_results, TRPO_files_name = All_seeds_rewards(TRPO_addresss,Environment_filter,True)
    dataframe = data_frame(PPO_files_results,PPO_files_name, DDPG_files_results,DDPG_files_name, SAC_files_results,\
                           SAC_files_name,TD3_files_results,TD3_files_name, TRPO_files_results, TRPO_files_name,None_attack_frame)
    Stack_plots(dataframe,save_path)
