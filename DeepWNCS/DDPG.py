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
from Util.utils import to_tensor
from copy import deepcopy
from model import ActorNetwork, CriticNetwork, DQNNetwork, Transition
from Util.huber_loss import HuberLoss
from Plant.pendulumSim import Pendulum
import Plant.pendulumParam as P

#Hyperparameters
lr_actor        = 0.0005
lr_critic       = 0.001
dtype = np.float32
    
class DDPG():
    def __init__(self, conf , device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device
        
        # initilize counter
        self.currIteration = 0
        
        # self.actor = ActorNetwork()
        # self.actor_target = deepcopy(self.actor)
        # self.critic = CriticNetwork()
        # self.critic_target = deepcopy(self.critic)
        
        # actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        # critic_optimizer  = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        
        
    
