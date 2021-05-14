# -*- coding: utf-8 -*-

'''
environments

'''
import matplotlib.pyplot as plt
import numpy as np
import Plant.pendulumParam as P
from Environments.Environment import environment
from Models.DQN import DQN
from Models.model import ReplayMemory, Transition
from Util.utils import to_tensor, set_cuda
import torch, argparse, pickle
from Plotter.trainDataPlotter import trainDataPlotter_DQN
from Plotter.testDataPlotter import testDataPlotter

# hyper parameter
dtype = np.float32
DQN_PATH = './LogData/model/dqn.pickle'
LOG_PATH = './LogData/log/DQN_log.pickle'

class MultipendulumSim_DQN():
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.is_cuda, self.device = set_cuda()
        self.agent = DQN(conf, self.device)

    def train(self, model_path, log_path):
        self.epi_durations, self.epi_rewards_mean, self.episode, self.step_count, self.epi_loss = [], [], [], [], []

        for epi in range(self.conf['num_episode']):  # episodes
            rewards = []
            self.episode.append(epi)
            step = 0
            epi_reward = 0.0
            print("--- episode %s ---"%(epi))
            state = self.env.reset()
            state_ts = to_tensor(state).reshape(-1).unsqueeze(0)
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    action = self.agent.select_action(state_ts)
                    next_state, reward, done, info = self.env.step(action.item(), t) # shape of next_state : [(2*num_plant) X 1]
                    done_mask = 0.0 if done else 1.0

                    next_state_ts = to_tensor(next_state).reshape(-1).unsqueeze(0)
                    reward_ts = to_tensor(np.asarray(reward).reshape(-1))
                    self.agent.memory.push_transition(state_ts, action, next_state_ts, reward_ts)
                    state_ts = next_state_ts
                    epi_reward += reward
                    step += 1
                    self.step_count.append(step)
                    rewards.append(reward)
                    if self.agent.memory.length() >= self.conf['memory_capacity']:
                        loss = self.agent.update()
                        self.epi_loss.append(loss)
                    if done:
                        break
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts


            if epi % self.conf['target_update'] == 0 and epi != 0:
                print("target update")
                self.agent.q_target.load_state_dict(self.agent.q.state_dict())
            self.epi_durations.append(epi_duration)
            self.epi_rewards_mean.append(epi_reward)
            print("epi_duration:", epi_duration)
            # print("epi_reward:%s, len:%s"%(epi_reward, len(rewards)))
            print("mean epi_reward:", epi_reward/len(rewards))
            if epi % 10 == 0 and epi != 0:
                torch.save(self.agent.q.state_dict(), model_path)
                self.save_log(log_path)
        torch.save(self.agent.q.state_dict(), model_path)
        self.save_log(log_path)

    '''
    def train(self):
        self.epi_rewards_mean = []
        self.epi_losses = []
        self.epi_durations = []
        self.episode = []
        self.step_count = []
        for epi in range(self.conf['num_episode']):  # episodes
            print("--- episode %s ---"%(epi))
            epi_reward = 0.0
            epi_duration = 0.0
            step = 0
            loss_list = []
            self.episode.append(epi)

            state = self.env.reset()                    # [2*num_plant, 1]
            state_ts = to_tensor(state).unsqueeze(0)    # [1, 2*num_plant, 1] # unsqueeze(0) on 'state' is necessary for reply memory

            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                t_next_plot = t + P.t_plot
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    action = self.agent.select_action(state_ts)     # action type: tensor [1X1]
                    next_state, reward, done, info = self.env.step(action.item(), t) # shape of next_state : [(2*num_plant) X 1]
                    epi_reward += reward
                    # self.env.step(0, t)    # test for env.step() function
                    if done:
                        # next_state_ts = None
                        # break
                        pass
                    else:
                        next_state_ts = to_tensor(next_state).unsqueeze(0)  # [1, 2*num_plant, 1]
                    reward_ts = to_tensor(np.asarray(reward).reshape(-1))   # it's size should be [1] for reply buffer

                    # memory push
                    if not done:
                        self.memory.push_transition(state_ts, action, next_state_ts, reward_ts)

                    state_ts = next_state_ts

                    currLoss = self.agent.optimization_model_v2(self.memory)   # model optimization step
                    if currLoss != None: loss_list.append(currLoss)
                    step += 1
                    self.step_count.append(step)
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts
                if next_state_ts == None:   # episode terminates
                    epi_duration = t
                    # break

            # episode done
            self.epi_rewards_mean.append(epi_reward/len(self.step_count))
            self.epi_durations.append(epi_duration)
            aver_loss = sum(loss_list)/len(loss_list)
            self.epi_losses.append(aver_loss)
            # The target network has its weights kept frozen most of the time
            if epi % self.conf['target_update'] == 0:
                self.agent.scheduler_target.load_state_dict(self.agent.scheduler.state_dict())
        # Save satet_dict
        torch.save(self.agent.scheduler.state_dict(), DQN_PATH)
        self.save_log()
    '''
    def test(self, env, model_path):
        test_actions = []
        # new agent
        new_agent = DQN(self.conf, self.device)
        new_agent.load_model(model_path)

        epi_reward = 0.0
        epi_duration = 0.0
        state = env.reset()                    # [2*num_plant, 1]
        state_ts = to_tensor(state).unsqueeze(0)    # [1, 2*num_plant, 1] # unsqueeze(0) on 'state' is necessary for reply memory
        # realtimePlot = testDataPlotter(self.conf)
        t = P.t_start
        while t < P.t_end:
            t_next_plot = t + P.t_plot
            epi_duration = t
            while t < t_next_plot:  # data plot period
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    action = new_agent.select_action(state_ts)     # action type: tensor [1X1]
                    next_state, reward, done, info = env.step(action.item(), t) # shape of next_state : [(2*num_plant) X 1]
                    epi_reward += reward
                    test_actions.append(action.item())
                    if done: break
                else:   # every 1 ms
                    env.update_plant_state(t) # plant status update
                t = t + P.Ts
            # self.update_dataPlot(realtimePlot, t, env) # update data plot
            if done: break
        # realtimePlot.print_totalError()
        # realtimePlot.plot_schedule(test_actions)

    def save_log(self, log_path):
        combined_stats = dict()
        combined_stats['episode/reward_mean'] = self.epi_rewards_mean
        combined_stats['episode/duration'] = self.epi_durations
        combined_stats['episode/dqn_loss'] = self.epi_loss
        combined_stats['episode/count'] = self.episode
        combined_stats['step/count'] = self.step_count
        with open(log_path, 'wb') as f:
            pickle.dump(combined_stats,f)


    def load_log(self, log_path):
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def update_dataPlot(self, dataPlot, t, env):
        r_buff, x_buff, u_buff = env.get_current_plant_status()
        for i in range(env.num_plant + 1): # +1 for wire network
            dataPlot.update(i, t, r_buff[i], x_buff[i], u_buff[i])
        dataPlot.plot()
        plt.pause(0.0001)

    def plot_cumulate_reward(self, log_path):
        '''
        plot log data on training results
        '''
        log_data = self.load_log(log_path)
        log_data_plotter = trainDataPlotter_DQN()
        log_data_plotter.plot(log_data)

if __name__ == '__main__':
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 5
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 2*num_plant + 1,
        'action_dim': num_plant+1,
        "num_episode": 1000,
        "memory_capacity": 1000,
        "batch_size": 32,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10,
        }
    pend_configuration = {}
    # amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # trigger_list = [100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100, 200, 300, 400, 500]  # ms
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [100, 200, 300, 400, 500]  # ms
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i],
                                             'trigger_time': trigger_list[i]}


    # create environments
    env = environment(num_plant, pend_configuration)

    # train
    multiPendulumSim_DQN = MultipendulumSim_DQN(env, configuration)
    multiPendulumSim_DQN.train(DQN_PATH, LOG_PATH)
    multiPendulumSim_DQN.plot_cumulate_reward(LOG_PATH)

    # test
    # new_env = environment(num_plant, pend_configuration, purpose = 'test')
    # multiPendulumSim_A2C.test(new_env, iteration=100, test_duration = 20, algorithm = 'A2C')
