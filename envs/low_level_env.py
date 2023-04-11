import torch
import random
import sys
from torch_geometric.nn import VGAE
import json
import pandas as pd
import copy
import numpy as np
import csv
import math

sys.path.insert(1, "../helpers")
from helpers.data_feature_gen import Featurizer

class Low_level_env:
    def __init__(self, ) -> None:
        df = pd.read_csv('data/sub_graphs_original.csv')
        self.curr_sub = json.loads(df.iloc[1][0])
    
    @property
    def action_shape(self):
        return len(self.curr_sub['nodes'])

    def sample_action(self):
        return random.randint(0, (self.action_shape - 1))

    def set_current_sub_vnr(self, sub_=None, vnr_=None, mapp=None):
        self.curr_sub = sub_
        self.curr_vnr = vnr_
        self.curr_map = mapp


    ##################### have to change the reward function ###################

    def get_reward(self, embeddable, sub_node):
        mult = 1 if embeddable else -1
        scaling_factor = 1
        unutil = sub_node['cpu'] / sub_node['cpu_max']
        print(unutil)
        return (mult * (1 - unutil) * scaling_factor)
    
    #############################################################
    
    def encode_graphs(self, sub_graph, vnr_graph, initial=False, ind=None):
        # only encode the sub_graph and just concatenate the current node features of vnr onto the encoding 
        sub_data_obj = Featurizer.make_data_obj(sub_graph)
        vnr_data_obj = Featurizer.make_data_obj(vnr_graph, sub=0)
        if initial == True and ind is None:
            ind = 0
        final_inp = [sub_data_obj, vnr_data_obj.x[ind]]
        return final_inp

    def reset(self, sub=None, sub_ind=None, vnr=None, map_skel=None):
        self.cum_reward = 0
        self.set_current_sub_vnr(sub_=sub, vnr_=vnr, mapp=map_skel)
        self.curr_map['sub'] = sub_ind
        # initial == True means that the first node of the current chosen VNR is selected for state repr
        init_state = self.encode_graphs(self.curr_sub, self.curr_vnr, initial=True)
        return init_state
    
    def step(self, action, ind_vnr=None):

        # this is done to take the sub state to the current one after 
        # all the previous mappings 
        temp_sub = self.temp_sub_change()

        sub_node = temp_sub['nodes'][action]
        vnr_node = self.curr_vnr['nodes'][ind_vnr]

        embeddable = False
        if(sub_node['cpu'] >= 0 and sub_node['mem'] >= 0):
            if(sub_node['cpu'] >= vnr_node['cpu']) and (sub_node['mem'] >= vnr_node['mem']):
                embeddable = True
                self.curr_map['vnr_node_ind'].append(ind_vnr)
                self.curr_map['sub_node_ind'].append(action)
                self.curr_map['cpu_mem'].append((vnr_node['cpu'], vnr_node['mem']))
        
        # this is done to obtain a new state if the embedding has taken place
        # otherwise it returns the same old state
        temp_sub = self.temp_sub_change()
        reward = self.get_reward(embeddable, sub_node)
        self.cum_reward += reward
        done = True if ind_vnr == (len(self.curr_vnr['nodes']) - 1) else False
        if not done:
            ind_vnr += 1
        # I have put the next states as None in both low and high models , so take care of that
        next_state = self.encode_graphs(temp_sub, self.curr_vnr, ind=ind_vnr)
        return next_state, reward, done, embeddable

    def temp_sub_change(self):
        temp_sub = copy.deepcopy(self.curr_sub)
        if len(self.curr_map['vnr_node_ind']) != 0 and len(self.curr_map['sub_node_ind']) != 0:
            # just take the recent mapping and change the substrate
            for x in range(len(self.curr_map['sub_node_ind'])):
                temp_sub['nodes'][self.curr_map['sub_node_ind'][x]]['cpu'] -= self.curr_map['cpu_mem'][x][0]
                temp_sub['nodes'][self.curr_map['sub_node_ind'][x]]['mem'] -= self.curr_map['cpu_mem'][x][1]
        return temp_sub
    
    def get_mappings(self):
        return self.curr_map
    
    def get_total_rew(self):
        return self.cum_reward / len(self.curr_vnr['nodes'])

    
