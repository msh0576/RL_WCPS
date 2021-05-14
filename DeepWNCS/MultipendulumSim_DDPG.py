# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:51:57 2020

@author: Sihoon
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle, torch
from copy import deepcopy
from Models.DDPG import DDPG
from Models.model import Actor, Critic
import Plant.pendulumParam as P
from Util.utils import to_tensor, set_cuda
from Environments.Environment import environment
from Environments.Wire_Environment import wire_environment
from Plotter.dataPlotter_v2 import dataPlotter_v2
from Plotter.trainDataPlotter import trainDataPlotter_DDPG
from itertools import count

path = '/home/sihoon/works/RL_WCPS-master/DeepWNCS/Inverted_Pendulum_Simulation'

DDPG_Log_ = path + '/LogData/log/DDPG_log_.pickle'
DDPG_Model_ = path + '/LogData/model/DDPG_model_.pickle'

# realtime plot iteration
REAL_TIME_PLOT = 100
class MultipendulumSim_DDPG():
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.is_cuda, self.device = set_cuda()
        self.agent = DDPG(conf, self.device)

    def train(self, model_path, log_path):
        self.epi_rewards, self.epi_durations, self.episode = [], [], []
        self.actor_losses, self.critic_losses = [], []
        self.agent.set_train()
        for epi in range(self.conf['num_episode']):  # episodes
            print("--- episode %s ---"%(epi))
            if epi % REAL_TIME_PLOT == 0:
                # dataPlot = dataPlotter_v2(self.conf)
                pass
            state = self.env.reset()
            # print("state:",state)
            step, epi_duaration, epi_reward = 0, 0, 0.
            t = P.t_start
            for t in count(start=P.t_start, step=P.Ts):        # one episode (simulation)
                epi_duration = t
                if t >= P.t_end:
                    break
                if round(t,3)*1000 % 10 == 0: # every 1 ms, schedule udpate
                    state_ts = to_tensor(state).reshape(-1).unsqueeze(0)    # [1, state_dim]
                    action = self.agent.select_action(state_ts) # numpy [batch_size, 1]
                    # print("t:", t)
                    next_state, reward, done, _ = self.env.step(action.item(), t)
                    if step == 0:
                        print("action with noise: {:0.3f}".format(action.item()))

                    # memory push
                    action_ts = to_tensor(action, requires_grad=False).squeeze(0)   # [1]
                    next_state_ts = to_tensor(next_state).reshape(-1).unsqueeze(0) # [1, state_dim]
                    reward_ts = to_tensor(reward)   # [1]
                    mask = to_tensor(np.asarray(done).reshape(-1))

                    self.agent.memory_push(state_ts, action_ts, reward_ts, next_state_ts, mask)

                    # update
                    step += 1
                    epi_reward += reward.item()
                    state = next_state

                    if done:
                        break
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update

            if self.agent.memory.length() > 1000: # warmup
                critic_loss, actor_loss = self.agent.update_policy_v2()
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)

            # an episode is done
            print("epi_duration: {:0.3f}, \t epi_reward: {:0.3f}, \t epi_reward_aver: {:0.3f}".format(epi_duration, epi_reward, epi_reward/step))
            if len(self.actor_losses) != 0:
                # print("epi_actor_loss average:", sum(self.actor_losses)/len(self.actor_losses))
                pass
            self.episode.append(epi)
            self.epi_durations.append(epi_duration)
            self.epi_rewards.append(epi_reward)
            if epi % 50 == 0:
                self.save_log(log_path)
        # Save satet_dict
        # torch.save(self.agent.actor.state_dict(), model_path)
        self.save_log(log_path)


    def update_dataPlot(self, dataPlot, t, env):
        '''
        shows current system states
        '''
        r, x, u= env.get_current_plant_status()
        dataPlot.update(i, t, r, x, u)
        dataPlot.plot()
        plt.pause(0.0001)

    def save_log(self, log_path):
        combined_stats = dict()
        combined_stats['episode/reward'] = self.epi_rewards
        # combined_stats['step/count'] = self.step_count
        combined_stats['episode/duration'] = self.epi_durations
        combined_stats['step/actor_loss'] = self.actor_losses
        combined_stats['step/critic_loss'] = self.critic_losses
        combined_stats['episode/count'] = self.episode
        with open(log_path, 'wb') as f:
            pickle.dump(combined_stats,f)


    def load_log(self, log_path):
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def plot_cumulate_reward(self, log_path):
        '''
        plot log data on training results
        '''
        log_data = self.load_log(log_path)
        log_data_plotter = trainDataPlotter_DDPG()
        log_data_plotter.plot(log_data)

if __name__ == '__main__':
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 1
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 1,
        "num_episode": 1500,
        "memory_capacity": 10000,
        "batch_size": 100,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10,
        "update_iteration": 200
        }
    pend_configuration = {}
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.01, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [10, 10, 10, 10, 10]  # ms
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i],
                                             'trigger_time': trigger_list[i]}


    env = wire_environment('wire', pend_configuration['pend_%s'%(0)])

    multiPendulumSim_DDPG = MultipendulumSim_DDPG(env, configuration)
    multiPendulumSim_DDPG.train(DDPG_Model_, DDPG_Log_)
    multiPendulumSim_DDPG.plot_cumulate_reward(DDPG_Log_)
