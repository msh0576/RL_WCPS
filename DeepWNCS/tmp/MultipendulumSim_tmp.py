# -*- coding: utf-8 -*-

'''
environments

'''
import matplotlib.pyplot as plt
import numpy as np
import pendulumParam as P
from Environment import environment
from DDPG import DDPG
from model import ReplayMemory, Transition
from utils import to_tensor
import torch, argparse, pickle
# for test
from pendulumSim import Pendulum
from dataPlotter_v2 import dataPlotter_v2

# hyper parameter
dtype = np.float32
MODEL_PATH = './LogData/model-state'
LOG_PATH = './LogData/'
LOG_FILE = 'log.pickle'

class MultipendulumSim():
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.set_cuda()
        
        self.agent = DDPG(conf, self.device)
        self.memory = ReplayMemory(conf)
        
        
        
        
    def train(self):
        self.epi_rewards = []
        self.epi_losses = []
        
        for epi in range(self.conf['num_episode']):  # episodes
            print("--- episode %s ---"%(epi))
            epi_reward = 0.0
            state = self.env.reset()                    # [2*num_plant, 1]
            state_ts = to_tensor(state).unsqueeze(0)    # [1, 2*num_plant, 1] # unsqueeze(0) on 'state' is necessary for reply memory
            dataPlot = dataPlotter_v2(self.conf)
            
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                t_next_plot = t + P.t_plot
                while t < t_next_plot:  # data plot period
                    if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                        
                        action = self.agent.select_action(state_ts)     # action type: tensor [1X1]
                        next_state, reward, done, info = self.env.step(action.item(), t) # shape of next_state : [(2*num_plant) X 1]
                        epi_reward += reward
                        # self.env.step(0, t)    # test for env.step() function
                        
                        if done: 
                            next_state_ts = None
                            break
                        else:
                            next_state_ts = to_tensor(next_state).unsqueeze(0)  # [1, 2*num_plant, 1]
                        reward_ts = to_tensor(np.asarray(reward).reshape(-1))   # it's size should be [1] for reply buffer
                        
                        # memory push
                        self.memory.push_transition(state_ts, action, next_state_ts, reward_ts)
                        
                        state_ts = next_state_ts
                        
                        # model optimization step
                        currLoss = self.agent.optimization_model(self.memory)
                    else:   # every 1 ms
                        self.env.update_plant_state(t) # plant status update
                    t = t + P.Ts
                # self.update_dataPlot(dataPlot, t) # update data plot
                if next_state_ts == None:   # episode terminates
                    dataPlot.close()
                    break
            
            # episode done
            self.epi_rewards.append(epi_reward)
            self.epi_losses.append(currLoss)
            # The target network has its weights kept frozen most of the time
            if epi % self.conf['target_update'] == 0:
                self.agent.scheduler_target.load_state_dict(self.agent.scheduler.state_dict())
    
        # Save satet_dict
        torch.save(self.agent.scheduler.state_dict(), MODEL_PATH)
        self.save_log()
        self.load_log()
        
    def save_log(self):
        combined_stats = dict()
        combined_stats['rollout/return'] = np.mean(self.epi_rewards)
        combined_stats['rollout/return_history'] = self.epi_rewards
        combined_stats['train/loss'] = self.epi_losses
        with open(LOG_PATH + LOG_FILE, 'wb') as f:
            pickle.dump(combined_stats,f)
        # combined-stats['train/loss_scheduler'] = 
        
        
        # combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        # combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        # combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        # combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        # combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        # combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        # combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        # combined_stats['total/duration'] = duration
        # combined_stats['total/steps_per_second'] = float(t) / float(duration)
        # combined_stats['total/episodes'] = episodes
        # combined_stats['rollout/episodes'] = epoch_episodes
        # combined_stats['rollout/actions_std'] = np.std(epoch_actions)
    
    def load_log(self):
        with open(LOG_PATH + LOG_FILE, 'rb') as f:
            data = pickle.load(f)
        print("data:", data)
    
    def set_cuda(self):
        self.is_cuda = torch.cuda.is_available()
        print("torch version: ", torch.__version__)
        print("is_cuda: ", self.is_cuda)
        print(torch.cuda.get_device_name(0))
        if self.is_cuda:
            self.device = torch.device("cuda:0")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            print("Program will run on *****CPU***** ")
            
    def update_dataPlot(self, dataPlot, t):
        r_buff, x_buff, u_buff = self.env.get_current_plant_status()
        for i in range(self.env.num_plant):
            dataPlot.update(i, t, r_buff[i], x_buff[i], u_buff[i])
        dataPlot.plot()
        plt.pause(0.0001)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_plant', type=int, default=1)
    
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    # args = parse_args()
    # print('args:', args)
    
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 1
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 2*num_plant,
        'action_dim': 2*num_plant,
        "num_episode": 2,
        "memory_capacity": 10000,
        "batch_size": 100,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 500
        }
    
    # create environments
    env = environment(num_plant)
    
    # train
    multiPendulumSim = MultipendulumSim(env, configuration)
    multiPendulumSim.train()
           
    

    '''
    # generate pendulum
    pend_1 = Pendulum(network = 'wireless', pend_id = 1)
    pend_2 = Pendulum(network = 'wire', pend_id = 2)
    
    # simulation data plot
    dataPlot = dataPlotter_v2()
    
    # main simulation loop
    t = P.t_start  # time starts at t_start
    while t < P.t_end:  # one episode
        _, r1, x1, u1 = pend_1.time_step(t)
        t, r2, x2, u2 = pend_2.time_step(t)
        
        
        dataPlot.update(1, t, r1, x1, u1)
        dataPlot.update(2, t, r2, x2, u2)
        dataPlot.plot()
        plt.pause(0.0001)
    
    # Keeps the program from closing until user presses a button.
    print('Press key to close')
    plt.waitforbuttonpress()
    '''