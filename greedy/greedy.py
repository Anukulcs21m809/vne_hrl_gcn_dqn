import sys
import pandas as pd
import json
from networkx.readwrite import json_graph
from networkx.algorithms import shortest_paths
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../data')
sys.path.insert(2, '../helpers')

from helpers.graph_gen import GraphGen
from helpers.utilities_ import Time_Sim

file_ =  pd.read_csv('data/sub_graphs_original.csv')

class Greedy_Selector:
    def __init__(self) -> None:
        self.sub_graphs = [json.loads(file_.iloc[i][0]) for i in range(3)]
        self.current_subs = copy.deepcopy(self.sub_graphs)
        self.map_skel = {'sub': None, 'vnr_node_ind': [], 'sub_node_ind': [
        ], 'cpu_mem': [], 'link_ind': [], 'bw': [], 'paths': [], 'dep_t': None}
        self.current_map = None
        self.embeddings = None
    
    def reset_map(self):
        self.current_map = copy.deepcopy(self.map_skel)

    # find the substrate graph with the largest available resources
    # return an index of the substrate
    def sub_with_largest_resources(self):
        sub_resources = np.zeros((3,2))
        mapp = copy.copy(self.map_skel)
        for x in range(len(self.current_subs)):
            for y in range(len(self.current_subs[x]['nodes'])):
                sub_resources[x][0] += self.current_subs[x]['nodes'][y]['cpu']
                sub_resources[x][1] += self.current_subs[x]['nodes'][y]['mem']
        mapp['sub']= np.argmax(sub_resources, axis=0)[0]
        self.current_map['sub'] = np.argmax(sub_resources, axis=0)[0]
        return np.argmax(sub_resources, axis=0)[0]
    
    # find the sub nodes that have the largest available cpu and memory 
    # return a list of sub node indexes
    def select_sub_nodes(self, sub_ind , vnr):
        only_sub_nodes = copy.copy(self.current_subs[sub_ind]['nodes'])
        only_sub_nodes = sorted(only_sub_nodes, key=lambda i: i['cpu'], reverse=True)
        sub_ind_list = [node['id'] for node in only_sub_nodes[:len(vnr['nodes'])]]
        self.current_map['vnr_node_ind'] = [node['id'] for node in vnr['nodes']]
        self.current_map['sub_node_ind'] = sub_ind_list
        return sub_ind_list
    
    # find the links using the nodes that have been chosen 
    # return mapping if embeddable otherwise return NONE
    def embed_nodes_and_links(self, vnr):
        node_embedded = False
        embed_cnt = 0
        for idx in range(len(self.current_map['sub_node_ind'])):
            vnr_node_idx = self.current_map['vnr_node_ind'][idx]
            sub_node_idx = self.current_map['sub_node_ind'][idx]
            sub_idx = self.current_map['sub']
            if self.current_subs[sub_idx]['nodes'][sub_node_idx]['cpu'] >= 0 and self.current_subs[sub_idx]['nodes'][sub_node_idx]['cpu'] >= vnr['nodes'][vnr_node_idx]['cpu']:
                self.current_map['cpu_mem'].append((vnr['nodes'][vnr_node_idx]['cpu'], vnr['nodes'][vnr_node_idx]['mem']))
                embed_cnt += 1
                node_embedded = True
            else:
                node_embedded = False
                break
        if node_embedded == False:
            return None
        else:
            link_embedded = False
            nx_sub = json_graph.node_link_graph(self.current_subs[self.current_map['sub']])
            reserved_paths = {}

            for v_link in vnr['links']:
                v_s_ind = v_link['source']
                v_t_ind = v_link['target']
                bw = v_link['bw']

                src = self.current_map['sub_node_ind'][self.current_map['vnr_node_ind'].index(v_s_ind)]
                tar = self.current_map['sub_node_ind'][self.current_map['vnr_node_ind'].index(v_t_ind)]

                path_ = shortest_paths.astar_path(nx_sub, src, tar)
                ind_of_links = []
                for x in range(len(path_) - 1):
                    i = 0
                    index_of_link = None
                    for link in self.current_subs[self.current_map['sub']]['links']:
                        if((link['source'] == path_[x] and link['target'] == path_[x+1]) or (link['source'] == path_[x+1] and link['target'] == path_[x])):
                            index_of_link = i
                            break
                        i += 1

                    if index_of_link is not None and index_of_link not in reserved_paths.keys():
                        reserved_paths[index_of_link] = self.current_subs[self.current_map['sub']]['links'][index_of_link]['bw']

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
                    self.current_map['link_ind'].append(ind_of_links)
                    self.current_map['paths'].append(path_)
                    self.current_map['bw'].append(bw)
            if link_embedded == False:
                return None
            else:
                return self.current_map
    
    def change_substrate(self, mappings):
        sub = copy.copy(self.current_subs[self.current_map['sub']])
        for x in range(len(mappings['sub_node_ind'])):
            sub['nodes'][mappings['sub_node_ind'][x]]['cpu'] -= mappings['cpu_mem'][x][0]
            sub['nodes'][mappings['sub_node_ind'][x]]['mem'] -= mappings['cpu_mem'][x][1]
        
        for x in range(len(mappings['link_ind'])):
            for y in mappings['link_ind'][x]:
                sub['links'][y]['bw'] -= mappings['bw'][x]
        self.current_subs[self.current_map['sub']] = sub
    
    def release_resources(self, resources):
        for resource in resources:
            sub_ind = resource['sub']
            sub = copy.deepcopy(self.current_subs[sub_ind])
            for x in range(len(resource['sub_node_ind'])):
                sub['nodes'][resource['sub_node_ind'][x]]['cpu'] += resource['cpu_mem'][x][0]
                sub['nodes'][resource['sub_node_ind'][x]]['mem'] += resource['cpu_mem'][x][1]
        
            for x in range(len(resource['link_ind'])):
                for y in resource['link_ind'][x]:
                    sub['links'][y]['bw'] += resource['bw'][x]
            
            self.current_subs[sub_ind] = sub
    
    def return_util(self):
        utils = []
        for sub in self.current_subs:
            for nodes in sub['nodes']:
                util = 1 - (nodes['cpu'] / nodes['cpu_max'])
                utils.append(util)
        return np.mean(utils)

# values_for_vnrs = [
#             [2,10], [10,20], [10,20], [15,20]
#         ]
# gr_gen = GraphGen(sub_nodes=30, max_vnr_nodes=10, prob_of_link=0.1, sub_values=None, vnr_values=values_for_vnrs)
# embedded_vnrs = []
# greed = Greedy_Selector()
# n_vnrs_generated = 0
# n_vnrs_embedded = 0
# x_ = []
# y_ = []
# time_sim = Time_Sim()

# for x in range(500):
#     greed.reset_map()

#     exceeded = True
#     resources_to_release = []
#     while exceeded:
#         if(len(embedded_vnrs) > 0):
#             # checking for the departure time of the
#             if(embedded_vnrs[0][0] <= x):

#                 vnr_to_release = embedded_vnrs[0]
#                 resources_to_release.append(vnr_to_release[1])

#                 embedded_vnrs = embedded_vnrs[1:]
#             else:
#                 exceeded = False
#         else:
#             exceeded = False
#     # this part is for releasing the resources
#     greed.release_resources(resources_to_release)
#     vnr = gr_gen.gen_rnd_vnr()
#     n_vnrs_generated += 1
#     sub_ind = greed.sub_with_largest_resources()
#     selected_sub_nodes = greed.select_sub_nodes(sub_ind, vnr)
#     # we get a None if the vnr is not embedded otherwise we get a mapping for the VNR
#     result = greed.embed_nodes_and_links(vnr)
#     if result is not None:
#         greed.change_substrate(result)
#         dep_t = time_sim.ret_dep_time()
#         embedded_vnrs.append((dep_t, result))
#         embedded_vnrs.sort(key = lambda i : i[0])
#         n_vnrs_embedded += 1
#     x_.append(x)
#     y_.append(n_vnrs_embedded/ n_vnrs_generated)

# plt.plot(x_, y_)
# plt.xlabel('n_vnrs_generated')
# plt.ylabel('acceptance_ratio')
# plt.show()