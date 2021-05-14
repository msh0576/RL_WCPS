# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:50:48 2020

@author: Sihoon
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random, collections, torch
from Util.utils import *
from copy import deepcopy
from Util.huber_loss import HuberLoss
from Plant.pendulumSim import Pendulum
import Plant.pendulumParam as P
from Models.model import Actor_DDPG, Critic_DDPG, Transition, ReplayMemory, Actor_DDPG_v2
from Common.memory import SequentialMemory
from Common.random_process import OrnsteinUhlenbeckProcess

#Hyperparameters
lr_actor        = 0.0005
lr_critic       = 0.001
dtype = np.float32

criterion = nn.MSELoss()

class DDPG():
    def __init__(self, conf , device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device
        self.batch_size = conf['batch_size']

        # initilize counter
        self.currIteration = 0

        self.actor = Actor_DDPG(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor_DDPG(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic_DDPG(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic_DDPG(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        # hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        # hard_update(self.critic_target, self.critic)

        # Create replay buffer
        # self.memory = SequentialMemory(limit=conf['memory_capacity'], window_length=1)
        self.memory = ReplayMemory(conf)
        self.random_process = OrnsteinUhlenbeckProcess(size=conf['action_dim'], theta=0.15, mu=0.0, sigma=0.1) # sigma 가 작을수록 random 값이 작아짐
        # Hyper-parameters
        self.is_training = True
        self.depsilon = 1e-5
        self.discount = 0.99
        self.epsilon = conf['epsilon_start']
        self.tau = 0.005
        self.step = 0
        if self.device == 'cuda:0':
            self.cuda()


    def set_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def select_action(self, state_ts, decay_epsilon=True):
        '''
            <output>
            action : [batch_size, 1], numpy
        '''
        '''
        # version: output action type is scalar
        action = to_numpy(
            self.actor(state_ts).detach()
        )
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -0.5, 0.5).squeeze(0)
        '''
        # version: output action type is tensor
        action = self.actor(state_ts)
        # self.actor.train()  # 왜 하는 거?
        # noise = self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        noise = np.random.normal(0, 0.5, size=self.conf['action_dim'])

        # if self.step % 1000 == 0:
        #     print("step: %s || epsilon: %s || noise: %s"%(self.step, self.epsilon, noise))
        action += to_tensor(noise)
        action = action.clamp(-0.5, 0.5)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.step += 1
        return to_numpy(action.detach())

    def reset(self):
        self.random_process.reset_states()

    def memory_push(self, state, action, reward, next_state, done):
        if self.is_training:
            self.memory.push_transition(state, action, next_state, reward, done)

    def update_policy(self):
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values
        # print("target_q_batch:", target_q_batch)

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        

        value_loss = criterion(q_batch, target_q_batch)
        # print("value_loss:", value_loss)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def update_policy_v2(self):
        for iter in range(self.conf['update_iteration']):
            transitions = self.memory.sample_batch(self.conf['batch_size'])
            # print("transitions:", transitions)
            one_batch = Transition(*zip(*transitions))

            # Get tensors from the batch
            state_batch = torch.cat(one_batch.state).to(self.device)    # [batch, state_dim]
            action_batch = torch.cat(one_batch.action).to(self.device).unsqueeze(1)  # [batch, 1]
            reward_batch = torch.cat(one_batch.reward).to(self.device).unsqueeze(1)  # [batch, 1]
            next_state_batch = torch.cat(one_batch.next_state).to(self.device)  # [batch, state_dim]
            done_batch = torch.cat(one_batch.done).to(self.device).unsqueeze(1)  # [batch,1]

            # Compute the target Q value
            next_state_action_values = self.critic_target([ next_state_batch, self.actor_target(next_state_batch)]) # [batch, 1]
            target_Q = reward_batch + ((1.0 - done_batch) * self.conf['gamma'] * next_state_action_values).detach() # [batch, 1]

            # Get current Q estimate
            current_Q = self.critic([ state_batch, action_batch ])
            print("self.actor(state_batch):", self.actor(state_batch))
            print("current_Q:", current_Q)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Compute actor loss
            actor_loss = -self.critic([ state_batch, self.actor(state_batch) ]).mean()
            # Optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()


            # print("<<4>>")
            # self.debug_grad(self.critic, 'fc1.weight')

            # Update the target networks
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()
    
    def update_parameters_for_MBPO(self, memory, batch_size, updates):
        '''
        the input of 'memory' is a (batch_state, batch_action, batch_reward, batch_next_state, batch_done) generated from the train_policy_repeats function
        so the batch_size of the (~) is same with env_batch_size (=12)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)                # [env_batch_size (=12), 5]
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)              # [env_batch_size , 1]
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # [env_batch_size , 1]
        done_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)     # [env_batch_size , 1]

        critic_loss, actor_loss = self.train(state_batch, action_batch, next_state_batch, reward_batch, done_batch)
        return critic_loss, actor_loss
        
    
    def train(self, state_batch, action_batch, next_state_batch, reward_batch, done_batch):
        '''
            <output>
            actor and critic loss
        '''
        # Compute the target Q value
        next_state_action_values = self.critic_target([ next_state_batch, self.actor_target(next_state_batch)]) # [batch, 1]
        target_Q = reward_batch + ((1.0 - done_batch) * self.conf['gamma'] * next_state_action_values).detach() # [batch, 1]

        # Get current Q estimate
        current_Q = self.critic([ state_batch, action_batch ])

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Compute actor loss
        actor_loss = -self.critic([ state_batch, self.actor(state_batch) ]).mean()
        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        # print("<<4>>")
        # self.debug_grad(self.critic, 'fc1.weight')

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    def set_train(self):
        """
        Sets the model in training mode
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def debug_grad(self, network, layer_name):
        for name, param in network.named_parameters():
            if name == layer_name:
                print("%s // grad:%s"%(name, param.grad))
    
    def Save_checkpoint(self, path, info):
        '''
            info: {epi_step, total_step}
        '''
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'epoch': 0,
            'loss': 0,
            'epi_step': info['epi_step'],
            'total_step': info['total_step']
        }, path)
    
    def Load_checkpoint(self, path, eval=False):
        checkpoint = torch.load(path)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        if eval == False:
            self.critic.train()
            self.critic_target.train()
            self.actor.train()
            self.actor_target.train()
        else:
            self.critic.eval()
            self.critic_target.eval()
            self.actor.eval()
            self.actor_target.eval()
    
    def Load_info(self, path):
        checkpoint = torch.load(path)
        return checkpoint['epi_step'], checkpoint['total_step']

criterion = nn.MSELoss()
class DDPG_for_PETS(DDPG):
    def __init__(self, args , device):
        super().__init__(args , device)
    
    def update_policy(self, state_batchs, action_batchs, next_state_batchs, reward_batchs, done_batchs):
        '''
            <input>
            state_batchs: list, [sample_size, horizon, 5]
            action_batchs: list, [sample_size, horizon, 1]
            next_state_batchs: list, [sample_size, horizon, 5]
            reward_batchs: list, [sample_size, horizon, 1]
            done_batchs: list, [sample_size, horizon, 1]
        '''
        for idx in range(len(state_batchs)):
            state_batch = torch.FloatTensor(state_batchs[idx]).to(self.device)    # [horizon, 5]
            action_batch = torch.FloatTensor(action_batchs[idx]).to(self.device)  # [horizon, 1]
            next_state_batch = torch.FloatTensor(next_state_batchs[idx]).to(self.device)    # [horizon, 5]
            reward_batch = torch.FloatTensor(reward_batchs[idx]).to(self.device)  # [horizon, 1]
            done_batch = torch.FloatTensor(done_batchs[idx]).to(self.device)                # [horizon, 1]

            critic_loss, actor_loss = super().train(state_batch, action_batch, next_state_batch, reward_batch, done_batch)
        return critic_loss, actor_loss


class DDPG_test():
    def __init__(self, observation_size, action_size, schedule_size, device):
        '''
        This algorithm is used to compare with a dreamer algorithm
            Args:
                action_size: environment action size: [schedule_size (1 + num_plant) + command_size (1 * num_plant)]:
        '''
        self.device = device
        self.actor = Actor_DDPG(observation_size, action_size, schedule_size).to(device=device)
        self.actor_target = Actor_DDPG(observation_size, action_size, schedule_size).to(device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic_DDPG(observation_size, action_size).to(device=device)
        self.critic_target = Critic_DDPG(observation_size, action_size).to(device=device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-4)


        
    
    def get_action(self, observation):
        '''
            Args:
                observation: tensor(GPU)
            Ruturns:
                action: tensor(GPU), [action_size, ]
        '''
        return self.actor(observation)

    def update_policy(self, memory, batch_size):
        '''
        when sample is appended to the memory, its configure should be follows:
            observation: tensor(CPU), [1, obs_size], 
            action: tensor(CPU), [1, action_size], 
            reward: tensor(CPU), [1, ], 
            done: tensor(CPU), [1,]
        '''
        gamma = 0.99
        tau = 0.005
        # Sample batch
        transitions = memory.sample_batch(batch_size)
        # print("transitions:", transitions)
        # print("*zip(*transitions):", *zip(*transitions))  # (state_1, ..., state_n) (action_1, ..., action_n) (next_state_1, ...) (reward_1, ...) (don_1, ...)
        one_batch = Transition(*zip(*transitions))

        # Get tensors from the batch
        observation_batch = torch.cat(one_batch.state).to(self.device)    # [batch, observation_size]
        action_batch = torch.cat(one_batch.action, dim=0).to(self.device)  # [batch, action_size]
        reward_batch = torch.cat(one_batch.reward).to(self.device).unsqueeze(1)  # [batch, 1]
        next_observation_batch = torch.cat(one_batch.next_state).to(self.device)  # [batch, observation_size]
        done_batch = torch.cat(one_batch.done).to(self.device).unsqueeze(1)  # [batch,1]

        # Compute the target Q value
        next_state_action_values = self.critic_target([ next_observation_batch, self.actor_target(next_observation_batch)]) # [batch, 1]
        target_Q = reward_batch + ((1.0 - done_batch) * gamma * next_state_action_values).detach() # [batch, 1]

        # Get current Q estimate
        current_Q = self.critic([ observation_batch, action_batch ])
        # print("self.actor(observation_batch):", self.actor(observation_batch))
        # print("current_Q:", current_Q)

        # Compute critic loss
        # critic_loss = F.mse_loss(current_Q, target_Q)
        critic_loss = criterion(current_Q, target_Q)

        # # Optimize the critic
        # print("here 0k 1")
        # self.critic_optim.zero_grad()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Compute actor loss
        actor_loss = -self.critic([ observation_batch, self.actor(observation_batch) ]).mean()
        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, tau)
        soft_update(self.critic_target, self.critic, tau)

        return critic_loss.item(), actor_loss.item()

    def set_eval_mode(self):
        self.critic.eval()
        self.critic_target.eval()
        self.actor.eval()
        self.actor_target.eval()
    
    def set_train_mode(self):
        self.critic.train()
        self.critic_target.train()
        self.actor.train()
        self.actor_target.train()