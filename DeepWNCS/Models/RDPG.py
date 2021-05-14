# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 08:58:46 2020

@author: Sihoon
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random, collections, torch
# from model import Actor_RDPG, Critic_RDPG
from Common.policy_networks import *
from Common.value_networks import *
from Util.utils import hard_update, to_tensor, to_numpy, soft_update
from Common.random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable

q_lr = 0.001
policy_lr = 0.0001

class RDPG:
    def __init__(self, conf, device):
        self.conf = conf
        self.state_space = conf['state_space']
        self.state_dim = self.state_space.shape[0]
        self.action_space = conf['action_space']
        self.action_dim = self.action_space.shape[0]
        self.hidden_dim = conf['hidden_dim']
        self.device = device

        # single-branch network structure as in 'Memory-based control with recurrent neural networks'
        self.qnet = QNetworkLSTM2(self.state_space, self.action_space, self.hidden_dim, num_plant=self.conf['num_plant']).to(self.device)
        self.target_qnet = QNetworkLSTM2(self.state_space, self.action_space, self.hidden_dim, num_plant=self.conf['num_plant']).to(self.device)
        self.policy_net = DPG_PolicyNetworkLSTM2(self.state_space, self.action_space, self.hidden_dim,
                                                 num_plant=self.conf['num_plant']).to(self.device)
        self.target_policy_net = DPG_PolicyNetworkLSTM2(self.state_space, self.action_space, self.hidden_dim,
                                                        num_plant=self.conf['num_plant']).to(self.device)

        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()

        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    def update(self, batch_size, replay_buffer, reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3, warmup=True):
        self.update_cnt+=1
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)
        # print("state len:%s, %s, %s"%(len(state), len(state[0]), len(state[0][0])))
        # print("state1 len:%s, %s, %s"%(len(state), len(state[1]), len(state[1][0])))
        # print('sample:', state, action,  reward, done)
        state      = torch.FloatTensor(state).to(self.device)   # shape: [batch X epi_len X state_dim]
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)  # shape: [batch X epi_len X action_dim]
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  # shape: [batch X epi_len X 1]
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

        # use hidden states stored in the memory for initialization, hidden_in for current, hidden_out for target
        predict_q, _ = self.qnet(state, action, last_action, hidden_in) # for q
        new_action, _ = self.policy_net.evaluate(state, last_action, hidden_in) # for policy
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out)  # for q
        predict_target_q, _ = self.target_qnet(next_state, new_next_action, action, hidden_out)  # for q

        predict_new_q, _ = self.qnet(state, new_action, last_action, hidden_in) # for policy. as optimizers are separated, no detach for q_h_in is also fine
        target_q = reward+(1-done)*gamma*predict_target_q # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()    # train qnet

        policy_loss = -torch.mean(predict_new_q)
        self.policy_optimizer.zero_grad()   # train policy_net

        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        policy_loss.backward(retain_graph=True)

        self.q_optimizer.step()
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt % target_update_delay==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()


    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()


'''
class RDPG:
    def __init__(self, conf, device):
        self.conf = conf
        self.state_dim = conf['state_dim']
        self.action_dim = conf['action_dim']
        self.device = device

        # create actor and critic network
        self.actor = Actor_RDPG(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor_RDPG(self.state_dim, self.action_dim).to(self.device)

        self.critic = Critic_RDPG(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic_RDPG(self.state_dim, self.action_dim).to(self.device)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=lr_rdpg)
        self.actor_optim  = optim.Adam(self.actor.parameters(), lr=plr_rdpg)

        #Create replay buffer
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=0.15, mu=0.0, sigma=0.2)
        # args.ou_theta:0.15 (noise theta), args.ou_sigma:0.2 (noise sigma), args.out_mu:0.0 (noise mu)


        self.epsilon = 1.0
        self.depsilon = 1.0/50000
        self.is_training = True
        self.tau = 0.001 # moving average for target network

    def random_action(self):
        action = np.random.uniform(0.,1.,self.action_dim)  # [-1,1] select as a number of action_dim
        return action

    def select_action(self, state, noise_enable=True, decay_epsilon=True):
        action, _ = self.actor(to_tensor(state).reshape(-1).unsqueeze(0))  # input shape = [batch X state_dim], action : type (tuple), shape [batch X action_dim]
        action = action.cpu().detach().numpy().squeeze(0)   # action shape [action_dim,]
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, 0., 1.)   # input 중 -1~1 을 벗어나는 값에 대해 -1 or 1 로 대체
        if decay_epsilon:
            self.epsilon -= self.depsilon

        return action

    def update_policy(self, memory, gamma=0.99):
        # Sample batch
        experiences = memory.sample(self.conf['batch_size'])    # type: list | shape: (max_epi_length(2000)-1 X batch(32) X 5(??))
        if len(experiences) == 0: # not enough samples
            return
        dtype = torch.cuda.FloatTensor

        policy_loss_total = 0
        value_loss_total = 0

        for t in range(len(experiences) - 1): # iterate over episodes
            target_cx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
            target_hx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)

            cx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)
            hx = Variable(torch.zeros(self.conf['batch_size'], 50)).type(dtype)

            # we first get the data out of the sampled experience
            # shape of state0, action, reward: [batch X state_dim X 1], [batch X 1], [batch X 1]
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t])) # batch 개수만큼 각 epi 중 t 시점에서 상태
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))

            target_action, (target_hx, target_cx) = self.actor_target(to_tensor(state1).reshape(self.conf['batch_size'],-1), (target_hx, target_cx))
            next_q_value = self.critic_target([
                to_tensor(state1).reshape(self.conf['batch_size'],-1),
                target_action
            ])

            target_q = to_tensor(reward) + gamma*next_q_value

            # Critic update
            current_q = self.critic([ to_tensor(state0).reshape(self.conf['batch_size'],-1), to_tensor(action) ])

            value_loss = F.smooth_l1_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss

            # Actor update
            action, (hx, cx) = self.actor(to_tensor(state0).reshape(self.conf['batch_size'],-1), (hx, cx))
            policy_loss = -self.critic([
                to_tensor(state0).reshape(self.conf['batch_size'],-1),
                action
            ])
            print("state0 require_grad:", to_tensor(state0).reshape(self.conf['batch_size'],-1).requires_grad)
            print("value_loss require_grad:", value_loss.requires_grad)
            print("1. policy_loss requires_grad:", policy_loss.requires_grad)
            policy_loss /= len(experiences) # divide by trajectory length
            policy_loss_total += policy_loss.mean()

            # update per trajectory
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            self.actor.zero_grad()
            policy_loss = policy_loss.detach().mean()
            print("2. policy_loss requires_grad:", policy_loss.requires_grad)
            policy_loss.backward()
            self.actor_optim.step()

            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
  '''
