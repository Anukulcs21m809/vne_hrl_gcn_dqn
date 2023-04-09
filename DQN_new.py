import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import copy

from envs.high_level_env import High_level_env
from envs.low_level_env import Low_level_env
# from helpers.data_feature_gen import Featurizer
from helpers.graph_gen import GraphGen
from helpers.utilities_ import ReplayMemory, Time_Sim
from neural_networks.higher_network import Higher_network
from neural_networks.lower_network import Lower_network
import sys

# sys.path.insert(1, "../data")
# sys.path.insert(2, '../envs')
# sys.path.insert(3, '../helpers')
# sys.path.insert(4, '../models')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN:
    def __init__(self, high_agent=1, max_vnr_nodes=8, n_sub_ftrs=6, n_vnr_ftrs=3, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4) -> None:
        self.high_agent = high_agent
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.learning_rate = lr
        
        self.max_vnr_nodes = max_vnr_nodes
        self.n_sub_ftrs = n_sub_ftrs
        self.n_vnr_ftrs = n_vnr_ftrs
        self.max_episodes = 10000
        self.max_time_steps_per_episode = 20000

        num_sub_graphs = 3
        num_nodes_sub = 20
        num_features_of_ffnn = (num_nodes_sub * num_sub_graphs) + self.max_vnr_nodes
        g1_h_ftrs = 2
        g1_o_ftrs = 1
        g2_h_ftrs = 2
        g2_o_ftrs = 1
        ff_hidden_size = 128
        n_classes_high = 3
        n_classes_low = 20

        # # cpu_range, mem_range, bandwidth_range
        values_for_subs = [
            [[50, 100], [64, 128], [50, 120]],
            [[60, 120], [50, 100], [60, 100]],
            [[80, 160], [64, 128], [50, 120]]
            ]

        # # number_of_nodes, cpu_req_range, mem_req_range, bandwidth_req_range
        values_for_vnrs = [
            [2,10], [10,20], [10,20], [15,20]
        ]

        self.graph_gen = GraphGen(sub_nodes=num_nodes_sub, max_vnr_nodes=self.max_vnr_nodes, prob_of_link=0.1, sub_values=values_for_subs, vnr_values=values_for_vnrs)
        self.sub_graphs_original = self.create_subs()

        if self.high_agent:
            self.env = High_level_env(sub_graphs=self.sub_graphs_original)#self.max_vnr_nodes, self.n_vnr_ftrs, self.n_sub_ftrs)
            self.policy_net = Higher_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, self.n_vnr_ftrs, g2_h_ftrs, g2_o_ftrs, ff_hidden_size, n_classes_high, num_features_of_ffnn)
            self.target_net = Higher_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, self.n_vnr_ftrs, g2_h_ftrs, g2_o_ftrs, ff_hidden_size, n_classes_high, num_features_of_ffnn)
            self.embedded_vnr_mappings = []
            self.map_skel = {'sub':None, 'vnr_node_ind':[], 'sub_node_ind':[], 'cpu_mem':[], 'link_ind':[], 'bw':[], 'paths':[], 'dep_t':None}           
        else:
            self.env = Low_level_env()
            self.policy_net = Lower_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, ff_hidden_size, n_classes_low, num_features_of_ffnn)
            self.target_net = Lower_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, ff_hidden_size, n_classes_low, num_features_of_ffnn)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)
    
        self.rew_buffer = deque([0.0], maxlen=1000)
        self.time_sim = Time_Sim()

        self.steps_done = 0

    # only use this function if we want to change the substrate graphs
    def create_subs(self):
        if self.high_agent:
            gr = self.graph_gen.make_cnst_sub_graphs()
        else:
            raise ValueError
        return gr
        
    def get_rnd_vnr(self):
        vnr = self.graph_gen.gen_rnd_vnr()
        return vnr

    def set_low_dqn(self, dqn_low):
        if self.high_agent:
            self.low_dqn = dqn_low
        else:
            raise ValueError
    
    def revenue_calc(self, revenues_, costs_):
        pass


    def release_embeddings(self, step, revenues_, costs_):
    # this part has been added to find all the vnrs that have expired
        exceeded = True
        resources_to_release = []
        while exceeded:
            if(len(self.embedded_vnr_mappings) > 0):
                # checking for the departure time of the 
                if(self.embedded_vnr_mappings[0][0] <= step):

                    vnr_to_release = self.embedded_vnr_mappings[0]
                    resources_to_release.append(vnr_to_release[1])

                    # for the revenue and cost calculation #########
                    revenue = 0
                    cost = 0
                    for cpu_mem in vnr_to_release[1]['cpu_mem']:
                        revenue = cpu_mem[0] + cpu_mem[1]
                        cost = copy.deepcopy(revenue)
                    for u in range(len(vnr_to_release[1]['paths'])):
                        revenue += vnr_to_release[1]['bw'][u]
                        cost += (vnr_to_release[1]['bw'][u] * (len(vnr_to_release[1]['paths'][u]) - 1))
                    
                    revenues_[vnr_to_release[1]['sub']] -= revenue
                    costs_[vnr_to_release[1]['sub']] -= cost
                    ########

                    self.embedded_vnr_mappings = self.embedded_vnr_mappings[1:]
                else:
                    exceeded = False
            else:
                exceeded = False
        # this part is for releasing the resources
        self.env.release_resources(resources_to_release)

        return revenues_, costs_
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.sample_action()]], device=device, dtype=torch.long)
        

    
    def high_loop(self):
        # the rnd vnr must be a json object which is passed to reset
        rnd_vnr = self.get_rnd_vnr()
        initial = True
        # high level state is given by the environement as [sub_1_data_obj, sub_2_data_obj, sub_3_data_obj, vnr_data_obj]
        # but while passing to the model, we need oto give it as ([sub_1_data_obj, sub_2_data_obj, sub_3_data_obj] , vnr_data_obj)
        state_high = self.env.reset(rnd_vnr)
        vnr = None
        self.time_sim.reset()
        ######################
        x_ = []
        y_ = []
        n_vnrs_generated = 0
        n_vnrs_embedded = 0
        revenues_ = [0, 0, 0]
        costs_ = [0, 0, 0]
        ######################

        arr_t = self.time_sim.ret_arr_time()
        cnt_ = 1
        prev_option = None
        for step in range(10):
            print("I am here")

            if step >= self.max_time_steps_per_episode:
                break

            revenues_ , costs_ =  self.release_embeddings(step, revenues_, costs_)

            if step >= arr_t:
                if initial == True:
                    vnr = rnd_vnr
                    initial = False
                
                n_vnrs_generated += 1
                revenue = 0
                cost = 0
                rev_cost_ratio_ = None

                high_option = self.select_action([state_high[:-1], state_high[-1]])

                print(high_option)

                arr_t = self.time_sim.ret_arr_time()







dqn = DQN()
dqn.create_subs()
dqn.high_loop()

        
        
    

