import numpy as np
import pickle, torch
import Plant.pendulumParam as P
from itertools import count
from Plotter.dataPlotter_v2 import dataPlotter_PETS
from Models.MPC import MPC, RandomPolicy
from Models.model import Ensemble_model_PETS
from Models.Agent import agent_PETS
from collections import deque
from Util.utils import save_checkpoint_dataset, load_checkpoint_dataset
import matplotlib.pyplot as plt


class MultipendulumSim_PETS():
    def __init__(self, conf, cuda_info, env, log_paths):
        self.args = conf
        self.is_cuda, self.device = cuda_info
        self.log_paths = log_paths
        self.agent = agent_PETS(self.args, env, self.device, self.log_paths)
        self.ensemble_model = Ensemble_model_PETS(self.args['network_size'], self.args['elite_size'], self.args['state_dim'], self.args['action_dim'], self.device)
        self.control_policy_CEM = MPC(self.args, self.ensemble_model, env, self.agent, use_random_planner=False)
        self.control_random_policy = RandomPolicy(self.args['action_dim'])

    

    def train(self, cont=False):
        state_buffer = deque(maxlen=40 * 10000)
        action_buffer = deque(maxlen=40 * 10000)
        next_state_buffer = deque(maxlen=40 * 10000)
        reward_buffer = deque(maxlen=40 * 10000)
        sum_reward_buffer = deque(maxlen=40 * 10000)
        total_step = 0
        train_iter = 0

        # perform model warmup
        if cont == False:
            state_buffer, action_buffer, next_state_buffer, reward_buffer, sum_reward_buffer, total_step = self.warmup_model(state_buffer, action_buffer, next_state_buffer, reward_buffer, sum_reward_buffer, total_step)
        else:
            self.ensemble_model.load_checkpoint(self.log_paths['model_path'])
            dataset = load_checkpoint_dataset(self.log_paths['dataset_path'])
            total_step = dataset['total_step']
            train_iter = dataset['train_iter']
            state_buffer = dataset['state_buffer']
            action_buffer = dataset['action_buffer']
            next_state_buffer = dataset['next_state_buffer']
            reward_buffer = dataset['reward_buffer']
            sum_reward_buffer = dataset['sum_reward_buffer']

        # training loop
        for iter in range(train_iter, self.args['num_train_iter']):
            print("########################################################")
            print("total_step :{}, train_iter:{}".format(total_step, iter))

            samples = []    # episodes sample
            for i in range(self.args['num_epi_per_iter']):
                print("-----episode {}-----".format(i))
                info = self.agent.sample(self.control_policy_CEM, total_step)
                samples.append(info)
                total_step = info['total_step']
                print("episode sum_reward: {}".format(info['sum_reward']))

            state_buffer.extend([sample['states'] for sample in samples])
            action_buffer.extend([sample['actions'] for sample in samples])
            next_state_buffer.extend([sample['next_states'] for sample in samples])
            reward_buffer.extend([sample['rewards'] for sample in samples])
            sum_reward_buffer.extend([sample['sum_reward'] for sample in samples])

            # train the dynamics model using the data
            self.control_policy_CEM.train(state_buffer, action_buffer, reward_buffer, next_state_buffer, epochs=5)

            # save checkpoint
            self.ensemble_model.save_checkpoint(self.log_paths['model_path'])
            dataset = {
                'state_buffer': state_buffer,
                'action_buffer': action_buffer,
                'next_state_buffer': next_state_buffer,
                'reward_buffer': reward_buffer,
                'sum_reward_buffer': sum_reward_buffer,
                'total_step': total_step,
                'train_iter': iter
            }
            save_checkpoint_dataset(dataset, self.log_paths['dataset_path'])

    def warmup_model(self, state_buffer, action_buffer, next_state_buffer, reward_buffer, sum_reward_buffer, total_step):
        samples = []
        for i in range(self.args['num_model_warmup']):
            print("----- model warmup {} -----".format(i))
            info = self.agent.sample(self.control_random_policy, total_step)
            samples.append(info)
            total_step = info['total_step']
            
        state_buffer.extend([sample['states'] for sample in samples])
        action_buffer.extend([sample['actions'] for sample in samples])
        next_state_buffer.extend([sample['next_states'] for sample in samples])
        reward_buffer.extend([sample['rewards'] for sample in samples])
        sum_reward_buffer.extend([sample['sum_reward'] for sample in samples])

        # train policy with initial samples
        self.control_policy_CEM.train(state_buffer, action_buffer, reward_buffer, next_state_buffer, epochs=5)
        return state_buffer, action_buffer, next_state_buffer, reward_buffer, sum_reward_buffer, total_step

    def evaluate(self):
        dataPlotter = dataPlotter_PETS(self.args)
        sum_reward = 0
        done = False

        # for epi in range(self.args['num_eval']):
        #     for t in count(start=P.t_start, step=P.Ts):
        #         # Finish of a simulation
        #         if t >= P.t_end:
        #             break
                
        #         # execute the environment
        #         _, _, _, reward, done = self.control_policy_CEM.sample(self.agent, t)
        #         sum_reward += reward

        #         # status plot
        #         if round(t,3)*1000 % 10 == 0: # every 10 ms
        #             print("t:",t)
        #             r, x, u = self.control_policy_CEM.get_plant_status()
        #             dataPlotter.update(pend_idx=0, t=t, reference=r, states=x, ctrl=u)
        #             dataPlotter.plot()
                
        #         if done:
        #             break
        dataPlotter.plt_waitforbuttonpress()

    def check_training_result(self):
        dataPlotter = dataPlotter_PETS(self.args)
        self.ensemble_model.load_checkpoint(self.log_paths['model_path'])
        dataset = load_checkpoint_dataset(self.log_paths['dataset_path'])

        reward_buffer = dataset['reward_buffer']
        reward_buffer = [epi_reward[:,:] for epi_reward in reward_buffer]
        reward_buffer = np.concatenate(reward_buffer, axis=0)

        sum_reward_buffer = dataset['sum_reward_buffer']
        sum_reward_buffer = [epi_sum_reward[:] for epi_sum_reward in sum_reward_buffer]
        sum_reward_buffer = np.concatenate(sum_reward_buffer, axis=0)
        sum_reward_x_axis = range(len(dataset['sum_reward_buffer']))
        dataPlotter.figure_sum_reward(sum_reward_x_axis, sum_reward_buffer)
        dataPlotter.plt_waitforbuttonpress()


