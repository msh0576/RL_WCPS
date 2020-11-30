# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:58:54 2020

@author: Sihoon
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from collections import namedtuple
import collections
import random
from torch.distributions import Categorical

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, config):
        self.config = config
        
        self.capacity = self.config['memory_capacity']
        self.memory = []
        self.position = 0
    
    def length(self):
        return len(self.memory)
    
    def push_transition(self, *args):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer
    
    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

# =============================================================================
# DQN
# =============================================================================

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
# =============================================================================
# A2C
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


# =============================================================================
# RDPG
# =============================================================================
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor_RDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Actor_RDPG, self).__init__()
        self.dtype = torch.cuda.FloatTensor
        
        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20, 50)
        self.lstm = nn.LSTMCell(50, 50)
        self.fc3 = nn.Linear(50, nb_actions)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.init_weights(init_w)

        self.cx = Variable(torch.zeros(1, 50)).type(self.dtype)
        self.hx = Variable(torch.zeros(1, 50)).type(self.dtype)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = Variable(torch.zeros(1, 50)).type(self.dtype)
            self.hx = Variable(torch.zeros(1, 50)).type(self.dtype)
        else:
            self.cx = Variable(self.cx.data).type(self.dtype)
            self.hx = Variable(self.hx.data).type(self.dtype)

    def forward(self, x, hidden_states=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if hidden_states == None:
            # x.shape = [batch, 50]
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.lstm(x, hidden_states)

        x = hx
        x = self.fc3(x)
        # x = self.tanh(x)
        x = self.sigm(x)
        return x, (hx, cx)

class Critic_RDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Critic_RDPG, self).__init__()
        
        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20 + nb_actions, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        #out = self.fc2(torch.cat([out,a],dim=1)) # dim should be 1, why doesn't work?
        out = self.fc2(torch.cat([out,a], 1)) # dim should be 1, why doesn't work? # 왜 여기서 action 과 합치지?
        out = self.relu(out)
        out = self.fc3(out)
        return out



# =============================================================================
# 
# =============================================================================

# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()
#         self.fc1 = nn.Linear(3, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc_mu = nn.Linear(64, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
#         return mu

# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()
        
#         self.fc_s = nn.Linear(3, 64)
#         self.fc_a = nn.Linear(1,64)
#         self.fc_q = nn.Linear(128, 32)
#         self.fc_3 = nn.Linear(32,1)

#     def forward(self, x, a):
#         h1 = F.relu(self.fc_s(x))
#         h2 = F.relu(self.fc_a(a))
#         cat = torch.cat([h1,h2], dim=1)
#         q = F.relu(self.fc_q(cat))
#         q = self.fc_3(q)
#         return q
