# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 08:53:44 2020

@author: Sihoon
"""

from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from Models.RDPG import RDPG
import torch, argparse, pickle
import Plant.pendulumParam as P
from Util.utils import to_tensor, set_cuda
from Plotter.trainDataPlotter import trainDataPlotter
from Plotter.testDataPlotter import testDataPlotter
from Common.memory import EpisodicMemory
from Environments.Environment import environment
from Common.buffers import *
# hyper parameter
dtype = np.float32

model_path = './LogData/rdpg.pickle'


class MultipendulumSim_RDPG:
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.max_episode_length = int(P.t_end//0.01)
        self.is_cuda, self.device = set_cuda()

        # self.replay_buffer =
        torch.autograd.set_detect_anomaly(True)
        self.agent = RDPG(conf, self.device)
        self.hidden_dim = self.conf['hidden_dim']
        self.replay_buffer = ReplayBufferLSTM2(self.conf['memory_capacity'])

    def train(self):
        rewards = []

        for epi in range(self.conf['num_episode']):  # episodes
            print("----episode %s----"%(epi))
            episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done = [], [], [], [], [], []
            q_loss_list, policy_loss_list = [], []
            state = self.env.reset()
            # print("initial state:", state)
            last_action = np.zeros(self.conf['action_space'].shape[0])
            hidden_out = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate

                    hidden_in = hidden_out
                    action, hidden_out = self.agent.policy_net.get_action(state, last_action, hidden_in)
                    schedule = self.env.action_to_schedule(action, self.conf['schedule_dim'])
                    # if round(t,3)*1000 % 10 == 0:
                    #     print("At time %s, action is %s, schedule is %s"%(t, action, schedule))
                    next_state, reward, done, _ = self.env.step(schedule, t)
                    if t == P.t_start:
                        ini_hidden_in = hidden_in
                        ini_hidden_out = hidden_out
                    episode_state.append(state)
                    episode_action.append(action)
                    episode_last_action.append(last_action)
                    episode_reward.append(reward)
                    episode_next_state.append(next_state)
                    episode_done.append(done)

                    # update
                    state = next_state
                    last_action = action

                    if len(self.replay_buffer) > self.conf['batch_size']:
                        for _ in range(self.conf['update_itr']):
                            q_loss, policy_loss = self.agent.update(self.conf['batch_size'], self.replay_buffer)
                            q_loss_list.append(q_loss)
                            policy_loss_list.append(policy_loss)

                    if done:
                        break
                    # if trajectory_step >= self.conf['trajectory_length']:
                    #     # self.agent.reset_lstm_hidden_state(done=False)    # why lstem become reset every its length?
                    #     trajectory_step = 0
                    #     if epi > self.conf['warmup']:
                    #         print("updat_policy")
                    #         self.agent.update_policy(self.memroy)
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts

            self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                               episode_reward, episode_next_state, episode_done)



            rewards.append(np.sum(episode_reward))
            if epi % 2 == 0:
                print("last reward:", rewards[-1])
                self.agent.save_model(model_path)



    def test(self, env, iteration):
        self.agent.load_model(model_path)
        for i in range(iteration):
            print("----iteration %s----"%(i))
            episode_reward = 0
            realtimePlot = testDataPlotter(self.conf)
            state = env.reset()
            last_action = np.zeros(self.conf['action_space'].shape[0])
            hidden_out = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cuda(),
                          torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)


            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                t_next_plot = t + P.t_plot
                while t < t_next_plot:  # data plot period
                    if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                        hidden_in = hidden_out
                        action, hidden_out = self.agent.policy_net.get_action(state, last_action, hidden_in)
                        schedule = env.action_to_schedule(action, self.conf['schedule_dim'])
                        next_state, reward, done, _ = env.step(schedule, t)

                        last_action = action
                        episode_reward += reward
                        # if done:
                        #     break
                    else:   # every 1 ms
                        env.update_plant_state(t) # plant status update
                    t = t + P.Ts
                self.update_dataPlot(realtimePlot, t, env) # update data plot
                if done: break

    def update_dataPlot(self, dataPlot, t, env):
        r_buff, x_buff, u_buff = env.get_current_plant_status()
        for i in range(env.num_plant + 1): # +1 for wire network
            dataPlot.update(i, t, r_buff[i], x_buff[i], u_buff[i])
        dataPlot.plot()
        plt.pause(0.0001)

if __name__ == '__main__':
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 3
    configuration = {
        'state_space': 0,
        'action_space': 0,
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 1,
        'hidden_dim': 64,
        'schedule_dim': 2*num_plant+1,
        "num_episode": 200,
        "memory_capacity": 10000,
        "batch_size": 3,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10,
        "trajectory_length": 100,
        'warmup': 1,
        'update_itr': 1
        }
    pend_configuration = {}
    amplitude_list = [0.1, 0.15, 0.25, 0.25]
    frequency_list = [0.1, 0.15, 0.15, 0.15]
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i]}




    # =============================================================================
    # RDPG
    # =============================================================================

    # train

    # create environments
    # env = environment(num_plant, pend_configuration)
    # configuration['state_space'] = env.observation_space
    # configuration['action_space'] = env.action_space

    # multiPendulumSim_RDPG = MultipendulumSim_RDPG(env, configuration)
    # multiPendulumSim_RDPG.train()

    new_env = environment(num_plant, pend_configuration, purpose = 'test')
    configuration['state_space'] = new_env.observation_space
    configuration['action_space'] = new_env.action_space
    multiPendulumSim_RDPG = MultipendulumSim_RDPG(new_env, configuration)
    multiPendulumSim_RDPG.test(new_env, iteration=100)
