# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:32:48 2020

@author: Sihoon
"""
import torch
class networkScheduler:
    def __init__(self, conf, device):
        self.bool = True
        self.conf = conf
        self.num_plant = self.conf['num_plant']
        self.device = device
        self.action = -1
    
    def generate_networkAction(self):
        '''
        if action = 1.0, then it means plant 1's sensor flow is selected
        if action = 1.5, then plant 1's actuation flow is selected
        '''
        if self.bool == True:
            self.bool = False
            action = 0.0                                                              
        else:
            self.bool = True
            action = 0.5
        return action
    
    
    def select_seqAction(self):
        '''
        schedule all flows sequentially
        '''
        self.action += 1
        if self.action % (self.conf['action_dim']) == 0:
            self.action = 0
        
        
        return torch.tensor(self.action, dtype=torch.float, device=self.device)