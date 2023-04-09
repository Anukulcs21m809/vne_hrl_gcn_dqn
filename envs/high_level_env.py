import torch
import random
import sys
import json
import pandas as pd
import copy
import torch.nn.functional as fn
from networkx.readwrite import json_graph
from networkx.algorithms import shortest_paths
import math
import numpy as np
import os

sys.path.insert(1, "../helpers")
# sys.path.insert(2, "../data")
from helpers.data_feature_gen import Featurizer

class High_level_env:
    def __init__(self, sub_graphs=None): #,max_vnr_nodes=None, vnr_in_ftr=None, sub_in_ftr=None) -> None:
        
        # this will always store the original state of the substrate graphs
        self.sub_graphs = sub_graphs

        # these sub graphs have to be instantiated at the DQN file
        # df = pd.read_csv('../data/sub_graphs_original.csv').reset_index()
        # for _, g in df.iterrows():
        #     G = json.loads(g['graphs'])
        #     self.sub_graphs.append(G)
        
    @property
    def action_shape(self):
        return len(self.sub_graphs)
    
    def sample_action(self):
        return random.randint(0, self.action_shape - 1)
    
    def choose_option(self, action):
        return copy.deepcopy(self.current_sub_state[action])
    
    # returns the data object for the current states for the substrate and the vnrs as a list [s1, s2, s3, v]
    def encode_graphs(self, vnr):
        encoded_data_objs = []
        for graph in self.current_sub_state:
            data_obj = Featurizer.make_data_obj(graph)
            encoded_data_objs.append(data_obj)
        encoded_data_objs.append(Featurizer.make_data_obj(vnr, sub=0))

        return encoded_data_objs

    # the vnr must be in json format so that the featurizer can make data obj out of it
    def reset(self, vnr=None):
        '''
        Returns: the observation of the initial state
        Reset the environment to the initial state so that a new episode (independent of the previous ones) may start
        '''
        self.not_embed_count = 0
        # resets the graphs to the original substrate graphs
        self.current_sub_state = copy.copy(self.sub_graphs)
        # incoming vnr is passed from the other file
        init_state = self.encode_graphs(vnr)
        return init_state
    
    def get_reward(self, link_embedded, mapp, vnr, option, cnt):
        mult = 1 if link_embedded else -1
        scaling_factor = 1
        const_ = (1 / (math.pow((cnt - 1), 2) + 1))
        rew_ = 1.0
        if link_embedded and mapp['link_ind'] is not None:
            for link in mapp['link_ind']:
                total_util = 0.0
                for ind_ in link:
                    sub_link = self.current_sub_state[option]['links'][ind_]
                    total_util += (1 - (sub_link['bw'] / sub_link['band_max']))
                rew_ += (total_util / (len(link) + 1))
        else:
            for link in vnr['links']:
                rew_ += link['bw']
            rew_ = (1 / rew_) * scaling_factor
        return (mult * const_ * rew_)
    
    def step(self, option, sub=None, vnr=None, link_embed=None, mapp=None, prev_vnr=None, cnt=None, rev_cost_ratio=None):
        
        self.not_embed_count += 1
        reward = self.get_reward(link_embed, mapp, prev_vnr, option, cnt)
        
        # replace the substrate that has been changed by the low agent
        self.current_sub_state[option] = sub
        done = True if self.not_embed_count > 5 else False
        next_state = self.encode_graphs(vnr) if not done else None
        
        return next_state, reward, done
    
    def get_utilization(self):
        node_utilization = np.zeros((3, len(self.sub_graphs[0]['nodes'])))
        link_utilization = []
        for x in range(len(self.current_sub_state)):
            link_utilization.append([0 for _ in range(len(self.current_sub_state[x]['links']))])

        for i in range(len(self.current_sub_state)):
            for j in range(len(self.current_sub_state[i]['nodes'])):
                node_utilization[i][j] = 1 - (self.current_sub_state[i]['nodes'][j]['cpu'] / self.current_sub_state[i]['nodes'][j]['cpu_max'])
            
            for k in range(len(self.current_sub_state[i]['links'])):
                link_utilization[i][k] = 1 - (self.current_sub_state[i]['links'][k]['bw'] / self.current_sub_state[i]['links'][k]['band_max'])
        
        return node_utilization, link_utilization
    
    def embed_link(self, sub, sub_ind, mappings, vnr):
        link_embedded = False
        nx_sub = json_graph.node_link_graph(sub)
        reserved_paths = {}

        if len(mappings['vnr_node_ind']) == 0 or len(mappings['sub_node_ind']) == 0:
            return link_embedded, mappings

        for v_link in vnr['links']:
            v_s_ind = v_link['source']
            v_t_ind = v_link['target']
            bw = v_link['bw']
            # embedded = False
            # embedded_cnt = 0

            src = mappings['sub_node_ind'][mappings['vnr_node_ind'].index(v_s_ind)]
            tar = mappings['sub_node_ind'][mappings['vnr_node_ind'].index(v_t_ind)]

            # since the substrate is a connected graph, we assume that path exists for each pair of nodes
            path_ = shortest_paths.astar_path(nx_sub, src, tar)

            ind_of_links = []
            for x in range(len(path_) - 1):
                i = 0
                index_of_link = None
                for link in sub['links']:
                    if((link['source'] == path_[x] and link['target'] == path_[x+1]) or (link['source'] == path_[x+1] and link['target'] == path_[x])):
                        index_of_link = i
                        break
                    i += 1
                
                if index_of_link is not None and index_of_link not in reserved_paths.keys():
                    reserved_paths[index_of_link] = sub['links'][index_of_link]['bw']

                if index_of_link is not None and reserved_paths[index_of_link] >= 0 and reserved_paths[index_of_link] >= bw:
                    ind_of_links.append(index_of_link)
                    
                    reserved_paths[index_of_link] -= bw
                    
                    link_embedded = True
                else:
                    link_embedded = False
                    break
            if link_embedded == False:
                break
            else:
                mappings['link_ind'].append(ind_of_links)
                mappings['paths'].append(path_)
                mappings['bw'].append(bw)

        return link_embedded, mappings
    
    def change_sub(self, sub, mappings):
        for x in range(len(mappings['sub_node_ind'])):
            sub['nodes'][mappings['sub_node_ind'][x]]['cpu'] -= mappings['cpu_mem'][x][0]
            sub['nodes'][mappings['sub_node_ind'][x]]['mem'] -= mappings['cpu_mem'][x][1]
        
        for x in range(len(mappings['link_ind'])):
            for y in mappings['link_ind'][x]:
                sub['links'][y]['bw'] -= mappings['bw'][x]
        return sub

    def release_resources(self, resources):
        for resource in resources:
            sub_ind = resource['sub']
            sub = copy.deepcopy(self.current_sub_state[sub_ind])
            for x in range(len(resource['sub_node_ind'])):
                sub['nodes'][resource['sub_node_ind'][x]]['cpu'] += resource['cpu_mem'][x][0]
                sub['nodes'][resource['sub_node_ind'][x]]['mem'] += resource['cpu_mem'][x][1]
        
            for x in range(len(resource['link_ind'])):
                for y in resource['link_ind'][x]:
                    sub['links'][y]['bw'] += resource['bw'][x]
            
            self.current_sub_state[sub_ind] = sub