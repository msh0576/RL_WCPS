# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:03:23 2020

@author: Sihoon
"""
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from Models.RDPG_v2 import RDPG_v2
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

model_path = './LogData/rdpg_v2.pickle'
log_path = './LogData/rdpg_v2_log.pickle'

class MultipendulumSim_RDPG_v2:
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.max_episode_length = int(P.t_end//0.01)
        self.is_cuda, self.device = set_cuda()

        # self.replay_buffer =
        torch.autograd.set_detect_anomaly(True)
        self.agent = RDPG_v2(conf, self.device)
        self.memory = EpisodicMemory(self.conf['memory_capacity'], self.max_episode_length, window_length=1)

    def train(self):
        rewards = []
        self.epi_durations, self.episode = [], []
        trajectory_step = 0

        for epi in range(self.conf['num_episode']):  # episodes
            print("----episode %s----"%(epi))
            self.episode.append(epi)
            episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done = [], [], [], [], [], []
            epi_duration = 0.
            state = self.env.reset()
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    action = self.agent.select_action(state, noise_enable=True)    # shape: [1], type: array
                    schedule = self.env.action_to_schedule(action, self.conf['schedule_dim'])
                    print("schedule:", schedule)
                    next_state, reward, done, _ = self.env.step(schedule, t)

                    # agent observe and update policy
                    self.memory.append(next_state, action, reward, done)
                    #update
                    trajectory_step += 1
                    episode_reward.append(reward)
                    state = next_state

                    if done:
                        self.agent.reset_lstm_hidden_state(done=True)
                        break

                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts
            if trajectory_step >= self.max_episode_length:   # traj_length should be larger than max_episode_length. reference to memory.py
                self.agent.reset_lstm_hidden_state(done=False)    # why lstem become reset every its length?
                trajectory_step = 0
                self.agent.update_policy(self.memory)
            print("episode duration:", epi_duration)
            print("trajectory_step:", trajectory_step)
            self.epi_durations.append(epi_duration)
            if epi % 10 == 0 and epi != 0:
                self.agent.save_model(model_path)
                self.save_log(log_path)
        self.agent.save_model(model_path)
        self.save_log(log_path)

    def test(self, env, iteration):
        self.agent.load_model(model_path)
        for i in range(iteration):
            print("----iteration %s----"%(i))
            episode_reward = 0
            realtimePlot = testDataPlotter(self.conf)
            state = env.reset()

            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                t_next_plot = t + P.t_plot
                while t < t_next_plot:  # data plot period
                    if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                        action = self.agent.select_action(state)    # shape: [1], type: array
                        schedule = env.action_to_schedule(action, self.conf['schedule_dim'])
                        next_state, reward, done, _ = env.step(schedule, t)
                        episode_reward += reward
                        if done:
                            break
                    else:   # every 1 ms
                        env.update_plant_state(t) # plant status update
                    t = t + P.Ts
                self.update_dataPlot(realtimePlot, t, env) # update data plot
                if done:
                    break


    def save_log(self, log_path):
        combined_stats = dict()
        combined_stats['episode/duration'] = self.epi_durations
        combined_stats['episode/count'] = self.episode
        with open(log_path, 'wb') as f:
            pickle.dump(combined_stats,f)

    def load_log(self, log_path):
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
        return data

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
        'state_space': 0,
        'action_space': 0,
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 1,
        'hidden_dim': 64,
        'schedule_dim': 2*num_plant+1,
        "num_episode": 1000,
        "memory_capacity": 100000,
        "batch_size": 5,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10,
        "trajectory_length": 10,
        'warmup': 1,
        'update_itr': 1
        }
    pend_configuration = {}
    amplitude_list = [0.1, 0.15, 0.2]
    frequency_list = [0.1, 0.15, 0.2]
    trigger_list = [100, 200, 300]  # ms
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i],
                                             'trigger_time': trigger_list[i]}


    # create environments
    env = environment(num_plant, pend_configuration)

    # =============================================================================
    # RDPG
    # =============================================================================

    # train
    multiPendulumSim_RDPG_v2 = MultipendulumSim_RDPG_v2(env, configuration)
    multiPendulumSim_RDPG_v2.train()


    # new_env = environment(num_plant, pend_configuration)
    # multiPendulumSim_RDPG_v2.test(new_env, iteration=100)
