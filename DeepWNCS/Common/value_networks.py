# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:38:45 2020

@author: Sihoon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from Common.initialize import *


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation, num_plant):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0] * num_plant
        else:  # high-dim state
            pass  
        self.activation = activation

    def forward(self):
        pass
    
class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation , num_plant):
        super().__init__( state_space, activation, num_plant)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]
    
class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, num_plant=1, output_activation=None):
        super().__init__(state_space, action_space, activation, num_plant)
        self.hidden_dim = hidden_dim
        
        self.linear1 = nn.Linear(self._state_dim+2*self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action, last_action], -1) 
        # print("x shape:", x.shape)
        x = self.activation(self.linear1(x))
        # print("linear1 x shape:", x.shape)
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   
    
    