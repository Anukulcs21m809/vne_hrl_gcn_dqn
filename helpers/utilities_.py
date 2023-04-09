from collections import namedtuple, deque
import random

# Transition = namedtuple('Transition', ('state', 'action' , 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action' , 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import numpy as np

class Time_Sim:
    def __init__(self) -> None:
        self.MU = 7 # mean arrival rate
        self.TAU = 0.3
        self.arrival_time = 0
    
    def ret_arr_time(self):
        u = np.random.uniform()
        self.arrival_time = self.arrival_time + (-np.log(1-u) / self.MU)
        arr_t = int(self.arrival_time * 10)
        return arr_t

    def ret_dep_time(self):
        u = np.random.uniform()
        dep_time = self.arrival_time + (-np.log(1-u) / self.TAU)
        dep_t = int(dep_time * 10)
        return dep_t
    
    def reset(self):
        self.arrival_time = 0
