import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model import (Actor, Critic)
from rpm import rpm
from random_process import *

from util import *

criterion = nn.MSELoss()


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.out = nn.Linear(21888, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.out(x)
        return x


class DDPG(object):
    def __init__(self, nb_status, nb_actions, args):
        self.num_actor = 3

        self.nb_status = nb_status * args.window_length
        self.nb_actions = nb_actions
        self.discrete = args.discrete
        self.pic = args.pic
        if self.pic:
            self.nb_status = args.pic_status
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'use_bn':args.bn
        }
        if args.pic:
            self.cnn = CNN(3, args.pic_status)
            self.cnn_optim = Adam(self.cnn.parameters(), lr=args.crate)
        self.actors = [Actor(self.nb_status, self.nb_actions) for _ in range(self.num_actor)]
        self.actor_targets = [Actor(self.nb_status, self.nb_actions) for _ in
                              range(self.num_actor)]
        self.actor_optims = [Adam(self.actors[i].parameters(), lr=args.prate) for i in range(self.num_actor)]

        self.critic = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        for i in range(self.num_actor):
            hard_update(self.actor_targets[i], self.actors[i])  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = rpm(args.rmsize) # SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = Myrandom(size=nb_actions)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.use_cuda = args.cuda
        # 
        if self.use_cuda: self.cuda()

    def normalize(self, pic):
        pic = pic.swapaxes(0, 2).swapaxes(1, 2)
        return pic

    def update_policy(self, train_actor = True):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_batch(self.batch_size)

        # Prepare for the target q batch
        if self.pic:
            state_batch = np.array([self.normalize(x) for x in state_batch])
            state_batch = to_tensor(state_batch, volatile=True)
            print('label 1')
            print('size = ', state_batch.shape)
            state_batch = self.cnn(state_batch)
            print('label 2')
            next_state_batch = np.array([self.normalize(x) for x in next_state_batch])
            next_state_batch = to_tensor(next_state_batch, volatile=True)
            next_state_batch = self.cnn(next_state_batch)
            next_q_values = self.critic_target([
                next_state_batch,
                self.actor_target(next_state_batch)
            ])
        else:
            index = np.random.randint(low=0, high=self.num_actor)
            next_q_values = self.critic_target([
                to_tensor(next_state_batch, volatile=True),
                self.actor_targets[index](to_tensor(next_state_batch, volatile=True)),
            ])
        # print('batch of picture is ok')
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount * to_tensor((1 - terminal_batch.astype(np.float))) * next_q_values

        # Critic update
        self.critic.zero_grad()
        if self.pic: self.cnn.zero_grad()

        if self.pic:
            state_batch.volatile = False
            q_batch = self.critic([state_batch, to_tensor(action_batch)])
        else:
            q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        # print(reward_batch, next_q_values*self.discount, target_q_batch, terminal_batch.astype(np.float))
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        if self.pic: self.cnn_optim.step()

        sum_policy_loss = 0
        for i in range(self.num_actor):
            self.actors[i].zero_grad()

            policy_loss = -self.critic([
                to_tensor(state_batch),
                self.actors[i](to_tensor(state_batch))
            ])

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            if train_actor:
                self.actor_optims[i].step()
            sum_policy_loss += policy_loss

            # Target update
            soft_update(self.actor_targets[i], self.actors[i], self.tau)

        soft_update(self.critic_target, self.critic, self.tau)

        return -sum_policy_loss / self.num_actor, value_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def cuda(self):
        for i in range(self.num_actor):
            self.actors[i].cuda()
            self.actor_targets[i].cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        self.memory.append([self.s_t, self.a_t, r_t, s_t1, done])
        self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        if self.discrete:
            return action.argmax()
        else:
            return action

    def select_action(self, s_t, decay_epsilon=True, return_fix=False, noise_level=0):
        actions = []
        status = []
        tot_score = []
        for i in range(self.num_actor):
            action = to_numpy(self.actors[i](to_tensor(np.array([s_t]), volatile=True))).squeeze(0)
            noise_level = noise_level * max(self.epsilon, 0)
            action = action + self.random_process.sample() * noise_level
            status.append(s_t)
            actions.append(action)
            tot_score.append(0.)

        scores = self.critic([to_tensor(np.array(status), volatile=True), to_tensor(np.array(actions), volatile=True)])
        for j in range(self.num_actor):
            tot_score[j] += scores.data[j][0]
        best = np.array(tot_score).argmax()

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = actions[best]
        return actions[best]

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_status()

    def load_weights(self, output, num=0):        
        if output is None: return
        for i in range(self.num_actor):
            actor = self.actors[i]
            actor_target = self.actor_targets[i]
            actor.load_state_dict(
                torch.load('{}/actor{}_{}.pkl'.format(output, num, i))
            )
            actor_target.load_state_dict(
                torch.load('{}/actor{}_{}.pkl'.format(output, num, i))
            )
        self.critic.load_state_dict(
            torch.load('{}/critic{}.pkl'.format(output, num))
        )
        self.critic_target.load_state_dict(
            torch.load('{}/critic{}.pkl'.format(output, num))
        )

    def save_model(self, output, num):
        if self.use_cuda:
            for i in range(self.num_actor):
                self.actors[i].cpu()
            self.critic.cpu()
        for i in range(self.num_actor):
            torch.save(
                self.actors[i].state_dict(),
                '{}/actor{}_{}.pkl'.format(output, num, i)
            )
        torch.save(
            self.critic.state_dict(),
            '{}/critic{}.pkl'.format(output, num)
        )
        if self.use_cuda:
            for i in range(self.num_actor):
                self.actors[i].cuda()
            self.critic.cuda()
