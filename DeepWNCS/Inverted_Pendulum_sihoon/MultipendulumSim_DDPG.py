# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:51:57 2020

@author: Sihoon
"""
import matplotlib.pyplot as plt
import numpy as np

from DDPG import DDPG
from model import Actor, Critic
import Plant.pendulumParam as P
from Util.utils import to_tensor, set_cuda
from Environment import environment
from Plotter.dataPlotter_v2 import dataPlotter_v2


class MultipendulumSim_DDPG():
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.is_cuda, self.device = set_cuda()
        self.agent = DDPG(conf, self.device)
    
    def train(self):
        
        for epi in range(self.conf['num_episode']):  # episodes
            realtimePlot = dataPlotter_v2(self.conf)
            state = self.env.reset()
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                t_next_plot = t + P.t_plot
                while t < t_next_plot:  # data plot period
                    self.env.wire_step(t)
                    t = t + P.Ts
                self.update_dataPlot(realtimePlot, t, self.env) # update data plot
    
    def update_dataPlot(self, dataPlot, t, env):
        r_buff, x_buff, u_buff = env.get_current_plant_status()
        for i in range(env.num_plant): # +1 for wire network
            dataPlot.update(i, t, r_buff[i], x_buff[i], u_buff[i])
        dataPlot.plot()
        plt.pause(0.0001)

if __name__ == '__main__': 
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 3
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 2*num_plant+1,
        "num_episode": 1,
        "memory_capacity": 10000,
        "batch_size": 32,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10
        }
    pend_configuration = {}
    amplitude_list = [0.1, 0.15, 0.25, 0.25]
    frequency_list = [0.1, 0.15, 0.15, 0.15]
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i]}
    
    
    # create environments
    env = environment(num_plant, pend_configuration)
    
    # train
    multiPendulumSim_DDPG = MultipendulumSim_DDPG(env, configuration)
    multiPendulumSim_DDPG.train()
    # multiPendulumSim_DDPG.plot_cumulate_reward()