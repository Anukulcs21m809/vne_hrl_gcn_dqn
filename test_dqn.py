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
import numpy as np
from statistics import mean
import wandb
import json
import pandas as pd

from envs.high_level_env import High_level_env
from envs.low_level_env import Low_level_env
# from helpers.data_feature_gen import Featurizer
from helpers.graph_gen_new import GraphGen
from helpers.utilities_ import ReplayMemory, Time_Sim
from neural_networks.higher_network import Higher_network
from neural_networks.lower_network import Lower_network
import sys
from greedy.greedy import Greedy_Selector

# sys.path.insert(1, "../data")
# sys.path.insert(2, '../envs')
# sys.path.insert(3, '../helpers')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wandb.init()

class DQN:
    def __init__(self, high_agent=1, max_vnr_nodes=10, n_sub_ftrs=6, n_vnr_ftrs=3, batch_size=256, gamma=0.99, eps_start=0.95, eps_end=0.05, eps_decay=500, tau=0.01, lr=1e-3, mu=0.2, value_for_vnr=None, depart_time=None) -> None:
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
        self.rew = 0.0

        num_sub_graphs = 3
        num_nodes_sub = 20
        num_features_of_ffnn_high = (num_nodes_sub * num_sub_graphs) + self.max_vnr_nodes
        num_features_of_ffnn_low = num_nodes_sub + self.n_vnr_ftrs
        g1_h_ftrs = 2
        g1_o_ftrs = 1
        g2_h_ftrs = 2
        g2_o_ftrs = 1
        ff_hidden_size = 128

        n_classes_high = 3
        n_classes_low = 20

        # # cpu_range, mem_range, bandwidth_range
        values_for_subs = [
        [16, 32, 48],
        [32, 64, 128],
        [256, 512, 1024]
        ]

        # # number_of_nodes, cpu_req_range, mem_req_range, bandwidth_req_range
        values_for_vnrs = [
            [3, 6],
            [2, 4],
            [2,4,8],
            [32, 48]
        ]if value_for_vnr is None else value_for_vnr

        max_running_time_high_agent = 2000
        max_not_embedded_threshold = 50 # this number cannot be very low as the substrates might be empty and it also cannot be very high because the equilibrium is already reached 

        if self.high_agent:
            self.graph_gen = GraphGen(sub_nodes=num_nodes_sub, max_vnr_nodes=self.max_vnr_nodes, prob_of_link=0.1, sub_values=values_for_subs, vnr_values=values_for_vnrs)
            file_ = pd.read_csv('models/model_to_eval_first/sub_graphs_used.csv')
            self.sub_graphs_original = [json.loads(file_.iloc[i][0]) for i in range(3)]
            self.env = High_level_env(sub_graphs=self.sub_graphs_original, max_running_time=max_running_time_high_agent, max_generated_cnt=max_not_embedded_threshold) #self.max_vnr_nodes, self.n_vnr_ftrs, self.n_sub_ftrs)
            self.policy_net = Higher_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, self.n_vnr_ftrs, g2_h_ftrs, g2_o_ftrs, ff_hidden_size, n_classes_high, num_features_of_ffnn_high)
            self.policy_net.load_state_dict(torch.load('models/model_to_eval_first/high_agent.pth'))
            self.target_net = Higher_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, self.n_vnr_ftrs, g2_h_ftrs, g2_o_ftrs, ff_hidden_size, n_classes_high, num_features_of_ffnn_high)
            self.target_net.load_state_dict(torch.load('models/model_to_eval_first/high_agent.pth'))
            self.map_skel = {'sub':None, 'vnr_node_ind':[], 'sub_node_ind':[], 'cpu_mem':[], 'link_ind':[], 'bw':[], 'paths':[], 'dep_t':None}           
        else:
            self.env = Low_level_env()
            self.policy_net = Lower_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, ff_hidden_size, n_classes_low, num_features_of_ffnn_low)
            self.policy_net.load_state_dict(torch.load('models/model_to_eval_first/low_agent.pth'))
            self.target_net = Lower_network(self.n_sub_ftrs, g1_h_ftrs, g1_o_ftrs, ff_hidden_size, n_classes_low, num_features_of_ffnn_low)
            self.target_net.load_state_dict(torch.load('models/model_to_eval_first/low_agent.pth'))
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        
        self.memory = ReplayMemory(10000)

        self.rew_buffer = deque([0.0], maxlen=10000)

        self.time_sim = Time_Sim(mu=mu, dep_t=depart_time)
        
    def get_rnd_vnr(self):
        vnr = self.graph_gen.gen_rnd_vnr()
        return vnr

    def set_low_dqn(self, dqn_low):
        if self.high_agent:
            self.low_dqn = dqn_low
        else:
            raise ValueError


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
    
    # the pair of sub_data and vnr_data is the state for both the high level and low level agent
    def select_action(self, sub_data, vnr_data):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(sub_data, vnr_data).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.sample_action()]], device=device, dtype=torch.long)
        
    def set_vals(self):
        self.steps_done = 0
        
    def high_loop(self):
        # the rnd vnr must be a json object which is passed to reset
        rnd_vnr = self.get_rnd_vnr()
        initial = True
        # high level state is given by the environement as [sub_1_data_obj, sub_2_data_obj, sub_3_data_obj, vnr_data_obj]
        # but while passing to the model, we need oto give it as ([sub_1_data_obj, sub_2_data_obj, sub_3_data_obj] , vnr_data_obj)
        state_high = self.env.reset(rnd_vnr)
        vnr = None
        self.time_sim.reset()
        arr_t = self.time_sim.ret_arr_time()

        self.low_dqn.set_vals()
        self.steps_done = 0
        
        cnt_ = 1
        prev_option = None
        self.embedded_vnr_mappings = []
        not_embed_count = 0
        cum_rev = 0
        ######################
        x_ = []
        y_ = []
        rev_cost_ = []
        cum_rev_ = []
        node_util = []
        n_vnrs_generated = 0
        n_vnrs_embedded = 0
        revenues_ = [0, 0, 0]
        costs_ = [0, 0, 0]
        ######################
        for step in count():

            if n_vnrs_generated > 1000:
                break

            # if cnt_ > 5: # if the same substrate is chosen more than 5 times, then end the episode
            #     break

            revenues_ , costs_ =  self.release_embeddings(step, revenues_, costs_)

            if step >= arr_t:
                if initial == True:
                    vnr = rnd_vnr
                    initial = False
                
                n_vnrs_generated += 1
                revenue = 0
                cost = 0
                rev_cost_ratio_ = None

                high_option_tensor = self.select_action(state_high[:-1], state_high[-1])
                # print(high_option_tensor)
                high_option = high_option_tensor.detach().item()

                sub = self.env.choose_option(high_option)

                prev_vnr = copy.deepcopy(vnr)
                if prev_option == high_option:
                    cnt_ += 1
                else:
                    cnt_ = 1
                prev_option = copy.copy(high_option)
                
                # the low level agent is used by the high level agent here
                node_embedded, node_mappings, cumulative_reward = self.low_dqn.low_loop(sub, high_option, vnr, map_skel=copy.deepcopy(self.map_skel), step_= step)
                
                ######################## from here to be continued

                link_embedded = False
                all_mappings = None

                if node_embedded:
                    link_embedded, all_mappings = self.env.embed_link(sub, high_option, node_mappings, vnr)
                    if link_embedded:
                        sub = self.env.change_sub(sub, all_mappings)
                        dep_t = self.time_sim.ret_dep_time()
                        self.embedded_vnr_mappings.append((dep_t, all_mappings))
                        self.embedded_vnr_mappings.sort(key = lambda i : i[0])
                        n_vnrs_embedded += 1

                        ##########################
                        # for the revenue and cost calculation
                        for cpu_mem in all_mappings['cpu_mem']:
                            revenue = cpu_mem[0] + cpu_mem[1]
                        cost = copy.deepcopy(revenue)
                        for u in range(len(all_mappings['paths'])):
                            revenue += all_mappings['bw'][u]
                            cost += (all_mappings['bw'][u] * (len(all_mappings['paths'][u]) - 1))
                        revenues_[all_mappings['sub']] += revenue
                        costs_[all_mappings['sub']] += cost
                        cum_rev += revenue
                        ###################################
                        
                    else:
                        not_embed_count += 1
                else:
                    not_embed_count += 1

                # if revenue != 0 and cost != 0:
                #     rev_cost_ratio_ = revenue / cost
                
                vnr = self.get_rnd_vnr()
                # the state is [s1_data_obj, s2_data_obj, s3_data_obj, vnr_data_obj]
                ##############################################
                next_state_high, reward_high, done_high = self.env.step(high_option, sub=sub, vnr=vnr, link_embed=link_embedded, 
                                                                        mapp=all_mappings, prev_vnr=prev_vnr, curr_time_step=step,
                                                                        cum_rew=cumulative_reward, cnt_=cnt_, generated_cnt=not_embed_count)
                ###############################################
                ###################################
                # for the node utilization calculation
                cpu_util_for_t = []

                for sub in self.env.current_sub_state:
                    for nodes in sub['nodes']:
                        cpu_util = 1 - (nodes['cpu'] / nodes['cpu_max'])
                        cpu_util_for_t.append(cpu_util)
                node_util.append(np.mean(cpu_util_for_t))
                ###################################
                # if done_high:
                #     next_state_high = None
                
                reward_high += cumulative_reward
                reward_high_tensor = torch.tensor([reward_high], device=device)

                self.memory.push(state_high, high_option_tensor, next_state_high, reward_high_tensor)
                self.rew_buffer.append(reward_high)

                state_high = next_state_high
                # self.optimize_model()

                # if done_high:
                #     print(n_vnrs_embedded)

                #     tot_rev_ratio = np.mean([0 if c == 0 else r / c for r, c in zip(revenues_, costs_)])
                            
                #     node_util, link_util = self.env.get_utilization()
                #     avg_node_util = np.mean(node_util, axis=1)
                #     avg_link_util = [mean(util) for util in link_util]
                #     wandb.log({'acceptance_rate': (n_vnrs_embedded/n_vnrs_generated), 'reward': np.mean(self.rew_buffer), 'avg_rev_cost_ratio' : tot_rev_ratio, 'avg_node_util_1' : avg_node_util[0], 'avg_node_util_2' : avg_node_util[1], 'avg_node_util_3' : avg_node_util[2], 'avg_link_util_1' : avg_link_util[0], 'avg_link_util_2' : avg_link_util[1],
                #                 'avg_link_util_3' : avg_link_util[2],})
                #     break

                tot_rev_ratio = np.mean([0 if c == 0 else r / c for r, c in zip(revenues_, costs_)])
                # print(step)
                # print(n_vnrs_embedded/n_vnrs_generated)
                # print('\n')
                x_.append(step)
                y_.append(n_vnrs_embedded/n_vnrs_generated)
                rev_cost_.append(tot_rev_ratio)
                cum_rev_.append(cum_rev)
                
                arr_t = self.time_sim.ret_arr_time()
        return x_, y_, rev_cost_, cum_rev_, node_util # node_util is a list of numbers
        
        # save both the trained models


    def low_loop(self, sub_, high_opt, vnr_, map_skel=None, step_=None):
        state_low = self.env.reset(sub=sub_, sub_ind=high_opt, vnr=vnr_, map_skel=map_skel)
        node_embedded = False
        prev_low_actions = []
        embed_cnt = 0 
        # self.steps_done = 0

        for x in range(len(vnr_['nodes'])):
            low_action = None
            prev_low_action = None
            while low_action in prev_low_actions or low_action == None:
                low_action_tensor = self.select_action(state_low[0], state_low[1])
                low_action = low_action_tensor.detach().item()
                if prev_low_action == low_action:
                    low_action = self.env.sample_action()
                prev_low_action = low_action
            prev_low_actions.append(low_action)
            

            # the next state can also be None so be careful
            next_state_low, reward_low, done_low, embedded = self.env.step(low_action, ind_vnr=x)

            reward_low_tensor = torch.tensor([reward_low], device=device)

            if done_low:
                next_state_low = None
            
            # in the original code, the state is a 2D tensor so that it can easily be concatenated
            # into a batch of 2D tensor states but in our case, our states are not tensors but data objects so, 
            # we have to handle the batch of transitions a little differently
            # maybe concatenate all the states into a list and then process them one by one 
            # and then concatenate the ouput tensor into a 2D tensor of Q values for each state 
            self.memory.push(state_low, low_action_tensor, next_state_low, reward_low_tensor)

            self.rew_buffer.append(reward_low)

            state_low = next_state_low
            if embedded:
                embed_cnt += 1

            # self.optimize_model()

            if done_low:
                break
        
        node_embedded = True if len(vnr_['nodes']) == embed_cnt else False
        cumulative_reward = self.env.get_total_rew()
        if node_embedded:
            map_skel = self.env.get_mappings()
        
        return node_embedded, map_skel, cumulative_reward
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None , batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]

        # this is only for a small batch size, for large batch size this error wont come
        if len(non_final_next_states) < 1:
            return

        state_batch = [s for s in batch.state]
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = []
        for x in range(self.batch_size):
            if self.high_agent:
                q_val = self.policy_net(state_batch[x][:-1], state_batch[x][-1])
            else:
                q_val = self.policy_net(state_batch[x][0], state_batch[x][1])
            q_values.append(q_val)


        q_values = torch.cat(q_values)
        state_action_values = q_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_q_values = []
            for x in range(len(non_final_next_states)):
                if self.high_agent:
                    next_q_val = self.target_net(non_final_next_states[x][:-1], non_final_next_states[x][-1])
                else:
                    next_q_val = self.target_net(non_final_next_states[x][0], non_final_next_states[x][1])
                next_q_values.append(next_q_val)
            next_q_values = torch.cat(next_q_values)
            next_state_values[non_final_mask] = next_q_values.max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.soft_update_weights()
    
    def soft_update_weights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]* self.tau + target_net_state_dict[key]*(1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

def evaluate(arr_rate, depart_time, resource_size, test_=None):
    dqn = DQN(mu=arr_rate, value_for_vnr=resource_size, depart_time=depart_time)
    low_dqn = DQN(high_agent = 0)
    dqn.set_low_dqn(low_dqn)
    x , y, rev_cost, cum_rev, node_util = dqn.high_loop()
    # plt.plot(x, y, color='r', label='hrl-dqn')
    # print('acceptance_ratio : ' , y[-1])
    # print('rev_cost : ', np.mean(rev_cost))
    # print('cumulative revenue : ', np.mean(cum_rev))
    # print('\n')
    f = open('result_text_files/results_dqn.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('dqn')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write(str('accept_ratio:{}'.format(str(y[-1]))) + '\n')
    f.write(str('rev_cost:{}'.format(str(np.mean(rev_cost)))) + '\n')
    f.write(str('cum_rev:{}'.format(str(np.mean(cum_rev)))) + '\n\n')
    f.close()

    f = open('result_text_files/util_results_dqn.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('dqn')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write('node util : {}'.format(node_util))
    f.write('\n\n')
    f.close()

    values_for_vnrs = resource_size
    gr_gen = GraphGen(sub_nodes=30, max_vnr_nodes=10, prob_of_link=0.1, sub_values=None, vnr_values=values_for_vnrs)
    embedded_vnrs = []
    greed = Greedy_Selector()
    n_vnrs_generated = 0
    n_vnrs_embedded = 0
    revenues_ = [0, 0, 0]
    costs_ = [0, 0, 0]
    x_ = []
    y_ = []
    utils_ = []
    rev_cost_ = []
    cum_rev_ = []
    time_sim = Time_Sim(mu=arr_rate, dep_t=depart_time)
    cum_rev = 0

    time_sim.reset()
    arr_t = time_sim.ret_arr_time()
    for step in count():
        greed.reset_map()

        if n_vnrs_generated > 1000:
            break

        exceeded = True
        resources_to_release = []
        while exceeded:
            if(len(embedded_vnrs) > 0):
                # checking for the departure time of the
                if(embedded_vnrs[0][0] <= step):

                    vnr_to_release = embedded_vnrs[0]
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

                    embedded_vnrs = embedded_vnrs[1:]
                else:
                    exceeded = False
            else:
                exceeded = False
        # this part is for releasing the resources
        greed.release_resources(resources_to_release)

        if step >= arr_t:

            vnr = gr_gen.gen_rnd_vnr()
            n_vnrs_generated += 1
            sub_ind = greed.sub_with_largest_resources()
            selected_sub_nodes = greed.select_sub_nodes(sub_ind, vnr)
            # we get a None if the vnr is not embedded otherwise we get a mapping for the VNR
            result = greed.embed_nodes_and_links(vnr)
            if result is not None:
                greed.change_substrate(result)
                dep_t = time_sim.ret_dep_time()
                embedded_vnrs.append((dep_t, result))
                embedded_vnrs.sort(key = lambda i : i[0])
                n_vnrs_embedded += 1

                ##########################
                # for the revenue and cost calculation
                for cpu_mem in result['cpu_mem']:
                    revenue = cpu_mem[0] + cpu_mem[1]
                cost = copy.deepcopy(revenue)
                for u in range(len(result['paths'])):
                    revenue += result['bw'][u]
                    cost += (result['bw'][u] * (len(result['paths'][u]) - 1))
                revenues_[result['sub']] += revenue
                costs_[result['sub']] += cost
                cum_rev += revenue
                ###################################
            arr_t = time_sim.ret_arr_time()
            # x_.append(x)
            utils_.append(greed.return_util())
            
            tot_rev_ratio = np.mean([0 if c == 0 else r / c for r, c in zip(revenues_, costs_)])

            y_.append(n_vnrs_embedded/ n_vnrs_generated)
            rev_cost_.append(tot_rev_ratio)
            cum_rev_.append(cum_rev)
            
    # print('acceptance_ratio : ' , y_[-1])
    # print('rev_cost : ', np.mean(rev_cost))
    # print('cumulative revenue : ', np.mean(cum_rev_))
    # print('\n')
    f = open('result_text_files/results_greedy.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('greedy')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write(str('accept_ratio:{}'.format(str(y_[-1]))) + '\n')
    f.write(str('rev_cost:{}'.format(str(np.mean(rev_cost_)))) + '\n')
    f.write(str('cum_rev:{}'.format(str(np.mean(cum_rev_)))) + '\n\n')
    f.close()

    f = open('result_text_files/util_results_greedy.txt', 'a')
    f.write('test:{}'.format(test_) + '\n')
    f.write(str('algo:{}'.format('greedy')) + '\n')
    f.write(str(arr_rate) + '\n')
    f.write(str(depart_time) + '\n')
    f.write(str(resource_size) + '\n')
    f.write('node util : {}'.format(utils_))
    f.write('\n\n')
    f.close()

arr_rates = [0.04 * x for x in range(1, 6)]
depart_times = [400+ (x * 100) for x in range(1, 6)]
resource_sizes = [
    [[3, 6], [2,4], [2,4,8], [32, 48]],
    [[3, 6], [4,6], [4,6,10], [36,52]],
    [[3, 6], [6,8], [6,8,12], [40,56]],
    [[3, 6], [8,10], [8,10,14], [44,60]],
    [[3, 6], [10,12], [10,12,16], [48,64]]
]

values = [arr_rates, depart_times, resource_sizes]

i = 0
for value in values:
    arr_r = 0.04
    dep_t = 600
    res_size = [[3, 6], [2,4], [2,4,8], [32, 48]]
    for val in value:    
        if i == 0:
            arr_r = val
            test = 'arr_rate'
        elif i == 1:
            dep_t = val
            test = 'dept_time'
        else:
            res_size = val
            test = 'vnr_size'
        evaluate(arr_r, dep_t, res_size, test_=test)
    i += 1


# plt.plot(x_, y_, color='g', label='greedy')
# plt.xlabel('n_vnrs_generated')
# plt.ylabel('acceptance_ratio')
# plt.legend()
# plt.show()
        
# we have to save the acceptance rate, rev/cost, revenue_gain in a text file
# for each metric and algo combo, create a separate file (e.g. dqn_acc_rate, greedy_rev_cost)



# some findings till now
# The steps should be reset for every episode because if for the high agent we dont do that, then the agent starts selecting only one option everytime (stops exploring)
# Same with that for the low agent but we reset its step for every episode, not every time low loop is run because the actions will be near random everytime if we do that        
    

