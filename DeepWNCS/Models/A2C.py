# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:13:33 2020

@author: Sihoon
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random, collections, torch
from Models.model import Actor, Critic
from Util.utils import to_tensor

class A2C:
    def __init__(self, conf, device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device


        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerC = optim.Adam(self.critic.parameters())


    def optimization_model(self, next_state, rewards, log_probs, values, masks):
        '''
        next_state : episode last state, which is used for G_t (return)
        rewards : a list that includes all rewards during an episode
        log_probs : a list that includes all log pi(a_t|s_t) during an episode, relative to actor network
        values : a list that includes all V(s_t), relative to critic network
        '''
        next_state_ts = to_tensor(next_state).reshape(-1) # [5*num_plant]
        next_value = self.critic(next_state_ts)     # V(s_{t+1}) 전체 분포?
        returns = self.compute_returns(next_value, rewards, masks)   # G_t = R + gamma * G_{t+1}

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values        # A = G_t - V(s_t)

        actor_loss = -(log_probs * advantage.detach()).mean() # -(log_pi(a|s) * A) 의 평균
        critic_loss = advantage.pow(2).mean()   # A^2 의 평균?

        # print("<<1>>")
        # self.debug_grad(self.critic, 'linear1.weight')
        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizerA.step()
        self.optimizerC.step()


        return actor_loss, critic_loss

    def optimization_model_v2(self, dist, action, state_ts, next_state_ts, reward, done, gamma=0.99):
        advantage = reward + (1-done) * gamma * self.critic(next_state_ts) - self.critic(state_ts)
        critic_loss = advantage.pow(2).mean()

        log_prob = dist.log_prob(action)   # scalar : log pi(a_t|s_t)
        actor_loss = -(log_prob * advantage.detach())   # [1]


        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizerA.step()
        self.optimizerC.step()

        return actor_loss, critic_loss

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def actor_load_model(self, path):
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def debug_grad(self, network, layer_name):
        for name, param in network.named_parameters():
            if name == layer_name:
                print("%s // grad:%s"%(name, param.grad))
