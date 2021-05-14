# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:03:38 2020

@author: Sihoon
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random, collections, torch
from model import Actor_RDPG, Critic_RDPG
from Util.utils import hard_update, to_tensor, to_numpy, soft_update
from random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable

q_lr = 0.001
policy_lr = 0.0001
class RDPG_v2:
    def __init__(self, conf, device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device
        
        # create actor and critic network
        self.actor = Actor_RDPG(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor_RDPG(self.state_dim, self.action_dim).to(self.device)
        
        self.critic = Critic_RDPG(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic_RDPG(self.state_dim, self.action_dim).to(self.device)
        
        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=q_lr)
        self.actor_optim  = optim.Adam(self.actor.parameters(), lr=policy_lr)
        
        #Create replay buffer
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=0.15, mu=0.0, sigma=0.2)
        # args.ou_theta:0.15 (noise theta), args.ou_sigma:0.2 (noise sigma), args.out_mu:0.0 (noise mu)
        
        
        self.epsilon = 1.0
        self.depsilon = 1.0/50000
        self.is_training = True
        self.tau = 0.001 # moving average for target network
    
    def random_action(self):
        action = np.random.uniform(0.,1.,self.action_dim)  # [-1,1] select as a number of action_dim
        return action
    
    def select_action(self, state, noise_enable=True, decay_epsilon=True):
        action, _ = self.actor(to_tensor(state).reshape(-1).unsqueeze(0))  # input shape = [batch(=1) X state_dim], action : type (tuple), shape [batch X action_dim]
        action = action.cpu().detach().numpy().squeeze(0)   # action shape [action_dim,]
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, 0., 1.)   # input 중 -1~1 을 벗어나는 값에 대해 -1 or 1 로 대체
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        return action
    
    def update_policy(self, memory, gamma=0.99):
        print("updating...")
        # Sample batch
        experiences = memory.sample(self.conf['batch_size'])    # type: list | shape: (max_epi_length(2000)-1 X batch(32) X 5(??))
        if len(experiences) == 0: # not enough samples
            return
        dtype = torch.cuda.FloatTensor
        
        policy_loss_total = 0
        value_loss_total = 0
        
        for t in range(len(experiences) - 1): # iterate over episodes
            # print("t:", t)
            target_cx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
            target_hx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
            
            cx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
            hx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
    
            # we first get the data out of the sampled experience
            # shape of state0, action, reward: [batch X state_dim], [batch X 1], [batch X 1]
            state0 = np.stack([trajectory.state0 for trajectory in experiences[t]]) # batch 개수만큼 각 epi 중 t 시점에서 상태만 추출
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
            action = np.stack([trajectory.action for trajectory in experiences[t]])
            reward = np.expand_dims(np.stack([trajectory.reward for trajectory in experiences[t]]), axis=1)
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
            state1 = np.stack([trajectory.state0 for trajectory in experiences[t+1]])
            
            target_action, (target_hx, target_cx) = self.actor_target(to_tensor(state1).reshape(self.conf['batch_size'],-1), (target_hx, target_cx))
            next_q_value = self.critic_target([
                to_tensor(state1).reshape(self.conf['batch_size'],-1),
                target_action
            ])
            
            target_q = to_tensor(reward) + gamma*next_q_value

            # Critic update
            current_q = self.critic([ to_tensor(state0).reshape(self.conf['batch_size'],-1), to_tensor(action) ])
            
            value_loss = F.smooth_l1_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss
            # update per trajectory
            self.critic.zero_grad()
            value_loss.backward()
            
            # Actor update
            action, (hx, cx) = self.actor(to_tensor(state0).reshape(self.conf['batch_size'],-1), (hx, cx))
            policy_loss = -self.critic([
                to_tensor(state0).reshape(self.conf['batch_size'],-1),
                action
            ])
            policy_loss /= len(experiences) # divide by trajectory length
            policy_loss_total += policy_loss.mean()
            policy_loss = policy_loss.mean()
            self.actor.zero_grad()
            policy_loss.backward()
            
            self.critic_optim.step()
            self.actor_optim.step()
            
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        print("update finish!")
    
    
    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)
        
    
    def save_model(self, path):
        torch.save(self.critic.state_dict(), path+'_q')
        torch.save(self.critic_target.state_dict(), path+'_target_q')
        torch.save(self.actor.state_dict(), path+'_policy')

    def load_model(self, path):
        self.critic.load_state_dict(torch.load(path+'_q'))
        self.critic_target.load_state_dict(torch.load(path+'_target_q'))
        self.actor.load_state_dict(torch.load(path+'_policy'))
        self.critic.eval()
        self.critic_target.eval()
        self.actor.eval()