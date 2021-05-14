# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:30:09 2020

@author: Sihoon
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random, collections, torch
from Util.utils import to_tensor
from copy import deepcopy
from Models.model import DQNNetwork, Transition, ReplayMemory
from Util.huber_loss import HuberLoss
from Plant.pendulumSim import Pendulum
import Plant.pendulumParam as P

#Hyperparameters
lr_dqn        = 0.0005
dtype = np.float32

class DQN():
    def __init__(self, conf , device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device

        self.q = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q_target = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.memory = ReplayMemory(self.conf)

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr_dqn)

        self.loss = HuberLoss()
        self.loss = self.loss.to(self.device)
        self.currIteration = 0

    def update(self):
        for i in range(1):
            if self.memory.length() < self.conf['batch_size']:
                return
            transitions = self.memory.sample_batch(self.conf['batch_size'])
            one_batch = Transition(*zip(*transitions))

            action_batch = torch.cat(one_batch.action).view(-1,1)   # [batch-size, 1]
            reward_batch = torch.cat(one_batch.reward).view(-1,1)   # [batch-size, 1]
            state_batch = torch.cat(one_batch.state).view(-1, self.conf['state_dim'])
            next_state_batch = torch.cat(one_batch.next_state).view(-1, self.conf['state_dim'])

            # dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

            # # compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # # columns of actions taken
            current_q = self.q(state_batch).gather(1, action_batch)

            # # compute V(s_{t+1}) for all next states and all actions,
            # # and we then take max_a { V(s_{t+1}) }
            next_q = self.q_target(next_state_batch).max(1)[0].view(-1,1)

            # # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            target_q = reward_batch + (self.conf['gamma'] * next_q)
            # print("current_q:%s, target_q:%s"%(current_q[0].item(), target_q[0].item()))

            # optimizer step
            loss = self.loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.cpu().item()

    ### epsilon-greedy algorithm ###
    def select_action(self, state_ts):
        '''
        input 'state_ts' is tensor [1, 2*num_plant, 1], which is unsqueezed type
        output 'action' is tensor [1X1]
        '''

        sample = random.random()
        eps_threshold = self.conf['epsilon_end'] + (self.conf['epsilon_start'] - self.conf['epsilon_end']) * math.exp(-1. * self.currIteration / self.conf['epsilon_decay'])
        self.currIteration += 1
        # if (self.currIteration % 1000) == 0:
        #     print("currIteration:%s, eps_threshold:%s"%(self.currIteration, eps_threshold))

        if sample > eps_threshold:
            with torch.no_grad():
                # argmax_a Q(s)
                action = self.q.forward(state_ts).argmax().view(1)     # .max(0)[1] : (0) 차원에서 최대 값[0]/index[1] and .view(1,1) : from tuple to [1X1]
                return action
        else:
            action = torch.tensor([random.randint(0, self.conf['action_dim']-1)], device = self.device, dtype=torch.long)     # [1X1]
            return action

'''
class DQN():
    def __init__(self, conf , device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device


        self.scheduler = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.scheduler_target = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        # define loss
        self.loss = HuberLoss()
        self.loss = self.loss.to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.scheduler.parameters(), lr = conf['learning_rate'])
        # Initialize target model
        self.scheduler_target.load_state_dict(self.scheduler.state_dict())
        self.scheduler_target.eval()

        # initilize counter
        self.currIteration = 0

        # self.actor = ActorNetwork()
        # self.actor_target = deepcopy(self.actor)
        # self.critic = CriticNetwork()
        # self.critic_target = deepcopy(self.critic)

        # actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        # critic_optimizer  = optim.Adam(self.critic.parameters(), lr=lr_critic)


    ### epsilon-greedy algorithm ###
    def select_action(self, state_ts):

        input 'state_ts' is tensor [1, 2*num_plant, 1], which is unsqueezed type
        output 'action' is tensor [1X1]

        sample = random.random()
        eps_threshold = self.conf['epsilon_end'] + (self.conf['epsilon_start'] - self.conf['epsilon_end']) * math.exp(-1. * self.currIteration / self.conf['epsilon_decay'])
        self.currIteration += 1
        if (self.currIteration % 1000) == 0:
            print("currIteration:%s, eps_threshold:%s"%(self.currIteration, eps_threshold))

        if sample > eps_threshold:
            with torch.no_grad():
                state_ts = state_ts.reshape((-1,)) # to [1 * (2*num_plant) * 1]
                # argmax_a Q(s)
                action = self.scheduler.forward(state_ts).max(0)[1].view(1,1)     # .max(0)[1] : (0) 차원에서 최대 값[0]/index[1] and .view(1,1) : from tuple to [1X1]
                return action
        else:
            action = torch.tensor([[random.randrange(self.conf['action_dim'])]], device = self.device, dtype=torch.long)     # [1X1]
            return action




    def optimization_model(self, memory):
        if memory.length() < self.conf['batch_size']:
            return
        # sample a batch
        transitions = memory.sample_batch(self.conf['batch_size'])
        one_batch = Transition(*zip(*transitions))

        # create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, one_batch.next_state)), device=self.device, dtype=torch.bool)  # tesor on GPU || [batch-size]
        non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None])   # tensor on GPU || [batch-size, 2*num_plant , 1]
        # print("non_final_mask: ", non_final_mask)
        # print("non_final_next_states shape: ", non_final_next_states.shape)

        # concatenate all batch elements into one
        action_batch = torch.cat(one_batch.action)   # [batch-size, 1] || tensor on GPU
        reward_batch = torch.cat(one_batch.reward)   # [batch-size]     || tensor on GPU
        state_batch = torch.cat(one_batch.state)     # [batch-size, 2*num_plant , 1]    || tensor on GPU

        # compute Q(s_t, a_t)
        state_batch = state_batch.reshape((self.conf['batch_size'], -1))            # [batch-size, 2*num_plant * 1]
        curr_state_values = self.scheduler(state_batch)                             # V(s_t)
        curr_state_action_values = curr_state_values.gather(1, action_batch)        # size: [batch-size, 1]

        # Get V(s_{t+1}) for all next states. By definition we set V(s)=0 if s is a terminal state.
        next_state_values = torch.zeros(self.conf['batch_size'], device = self.device)
        num_true = (non_final_mask == True).sum(dim=0)
        non_final_next_states = non_final_next_states.reshape((num_true, -1))
        next_state_values[non_final_mask] = self.scheduler_target(non_final_next_states).max(1)[0].detach()     # size: [batch-size] || max_a Q[nextState , , theta-]  ||  slicing the tensor using boolean list

        # Get the expected Q values
        expected_state_action_values = reward_batch + (self.conf['gamma'] * next_state_values)        # size: [batch-size]

        # compute loss: temporal difference error
        loss = self.loss(curr_state_action_values, expected_state_action_values.unsqueeze(1))       # both of size: [batch-size, 1]
        # 이게 점점 줄어들어야 되는데...

        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def optimization_model_v2(self, memory):
        if memory.length() < self.conf['batch_size']:
            return
        # batch = memory.sample(self.conf['batch_size'])
        transitions = memory.sample_batch(self.conf['batch_size'])
        one_batch = Transition(*zip(*transitions))

        action_batch = torch.cat(one_batch.action)   # [batch-size, 1]
        reward_batch = torch.cat(one_batch.reward).view(-1,1)   # [batch-size, 1]
        state_batch = torch.cat(one_batch.state).view(-1, self.conf['state_dim'])
        next_state_batch = torch.cat(one_batch.next_state).view(-1, self.conf['state_dim'])

        # dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # # compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # # columns of actions taken
        current_q = self.scheduler(state_batch).gather(1, action_batch)

        # # compute V(s_{t+1}) for all next states and all actions,
        # # and we then take max_a { V(s_{t+1}) }
        next_state_action_values = self.scheduler(next_state_batch).detach()
        next_q = next_state_action_values.max(1)[0].view(-1,1)

        # # compute target q by: r + gamma * max_a { V(s_{t+1}) }
        target_q = reward_batch + (self.conf['gamma'] * next_q)

        # optimizer step
        loss = self.loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def load_model(self, path):
        self.scheduler = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.scheduler.load_state_dict(torch.load(path))
        self.scheduler.eval()
'''
