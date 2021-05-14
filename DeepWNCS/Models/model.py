# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:58:54 2020

@author: Sihoon
"""
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from collections import namedtuple
import collections
import random
from torch.distributions import Categorical
from Util.utils import to_tensor, to_numpy

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, config):
        self.config = config

        self.capacity = self.config['memory_capacity']
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, *args):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

# Transition_v2 = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory_test(object):
    '''
    for comparing with dreamer algorithm
    '''
    def __init__(self, memory_capacity):
        self.capacity = memory_capacity
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def append(self, *args):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

class ReplayMemory_SAC:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# =============================================================================
# DQN
# =============================================================================

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# =============================================================================
# A2C
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))   # categorical은 output들의 확률분포를 만들어내는 함수같음
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


# =============================================================================
# RDPG
# =============================================================================
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor_RDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Actor_RDPG, self).__init__()
        self.dtype = torch.cuda.FloatTensor

        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20, 50)
        self.lstm = nn.LSTMCell(50, 50)
        self.fc3 = nn.Linear(50, nb_actions)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.init_weights(init_w)

        self.cx = Variable(torch.zeros(1, 50)).type(self.dtype)
        self.hx = Variable(torch.zeros(1, 50)).type(self.dtype)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = Variable(torch.zeros(1, 50)).type(self.dtype)
            self.hx = Variable(torch.zeros(1, 50)).type(self.dtype)
        else:
            self.cx = Variable(self.cx.data).type(self.dtype)
            self.hx = Variable(self.hx.data).type(self.dtype)

    def forward(self, x, hidden_states=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if hidden_states == None:
            # x.shape = [batch, 50]
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.lstm(x, hidden_states)

        x = hx
        x = self.fc3(x)
        # x = self.tanh(x)
        x = self.sigm(x)
        return x, (hx, cx)

class Critic_RDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Critic_RDPG, self).__init__()

        self.fc1 = nn.Linear(nb_states, 20)
        self.fc2 = nn.Linear(20 + nb_actions, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        #out = self.fc2(torch.cat([out,a],dim=1)) # dim should be 1, why doesn't work?
        out = self.fc2(torch.cat([out,a], 1)) # dim should be 1, why doesn't work? # 왜 여기서 action 과 합치지?
        out = self.relu(out)
        out = self.fc3(out)
        return out



# =============================================================================
# DDPG
# =============================================================================

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor_DDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor_DDPG, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        out = out
        return out

class Actor_DDPG_v2(nn.Module):
    '''
        <model's input>
            state: [5 * num_plant,]
        <model's output>
            is schedule + control command
            the schedule: , [num_plant + 1, ]
            the control command: tensor.float, [1 * num_plant,]: (-1 ~ 1)
    '''
    def __init__(self, nb_states, nb_actions, schedule_size, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor_DDPG_v2, self).__init__()
        import math
        self.schedule_size = schedule_size
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        '''
            <Return>
                output[:][:schedule_size]: schedule index distribution
                output[:][schedule_size:] are control commands
        '''
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        # print("out shape:", out.shape)
        schedule_dist = F.softmax(out[:][:self.schedule_size], dim=0)
        command = out[:][self.schedule_size:]
        out = torch.cat((schedule_dist, command), dim=0)
        return out

class Critic_DDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic_DDPG, self).__init__()
        self.fc1 = nn.Linear(nb_states + nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.relu(self.fc1(torch.cat([x,a],1)))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# =============================================================================
# PETS
# =============================================================================
class PENN(nn.Module):
    '''
        (P)robabilistic (E)nsemble of (N)eural (N)etworks
    '''
    def __init__(self, state_dim, action_dim, device, hidden_size=200, learning_rate=1e-2):
        super(PENN, self).__init__()
        
        self.nn1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_dim = state_dim
        # Add mean and variance
        self.nn4 = nn.Linear(hidden_size, self.output_dim * 2)

        # Log variange bounds
        self.max_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 3, requires_grad=True).to(device)
        self.min_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 7, requires_grad=True).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    
    def forward(self, x):
        '''
            <output>
            mean, var of the next state
        '''

        nn1_output = self.nn1(x)
        nn2_output = self.nn2(nn1_output)
        nn3_output = self.nn3(nn2_output)
        nn4_output = self.nn4(nn3_output)

        mean = nn4_output[:, :self.output_dim]
        raw_var = nn4_output[:, self.output_dim:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_var)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar
    
    def loss(self, mean, logvar, targets):
        '''
            mse_loss, var_loss  : scalar value
        '''
        var = torch.exp(logvar)
        inv_var = torch.exp(-logvar)
        
        norm_output = mean - targets
        # Calculate loss: Mahalanobis distance + log(det(cov))
        # mse_loss = torch.mean(torch.pow(mean - targets, 2) * inv_var)
        # var_loss = torch.mean(logvar)
        # total_loss = mse_loss + var_loss

        loss = torch.mul(torch.mul(norm_output, inv_var), norm_output)
        loss = torch.sum(loss, 1)
        loss += torch.log(torch.prod(var, axis=1))
        loss = torch.mean(loss)
        return loss
    
    def model_train(self, loss):
        self.optimizer.zero_grad()
        # loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()



def get_indices(n):
    return np.random.choice(range(n), size=n, replace=True)

class Ensemble_model_PETS():
    def __init__(self, network_size, elite_size, state_dim, action_dim, device, reward_dim=1):
        self.network_size = network_size
        self.elite_size = elite_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.model_list = []
        self.elite_model_idxes = []
        for _ in range(network_size):
            self.model_list.append(PENN(state_dim, action_dim, device).to(device))
    
    def predict(self, states, actions, idxs):
        '''
            At a time step, predict "a next_state" for each ensemble model
            <Argument>
                states: np.ndarray, [batch, state_dim]
                actions: np.ndarray, [batch, 1]
                idxs: list, [pop_size * max_iters = 250]
            <Return>
                ensemble_mean: np.ndarray, [network_size, batch, state_dim]
                ensemble_logvar: np.ndarray, [network_size, batch, state_dim]
        '''
        next_states = np.zeros_like(states)
        inputs = np.concatenate((states, actions), axis=1)
        assert inputs.shape[1] == (self.action_dim + self.state_dim)

        ensemble_mean = np.zeros((self.network_size, inputs.shape[0], self.state_dim))
        ensemble_logvar = np.zeros((self.network_size, inputs.shape[0], self.state_dim))
        # for i in range(0, inputs.shape[0], batch_size):
        #     input = to_tensor(inputs[i:min(i + batch_size, inputs.shape[0])])
        #     for idx in range(self.network_size):
        #         pred_2d_mean, pred_2d_logvar = self.model_list[idx](input)
        #         ensemble_mean[idx,i:min(i + batch_size, inputs.shape[0]),:], ensemble_logvar[idx,i:min(i + batch_size, inputs.shape[0]),:] \
        #             = to_numpy(pred_2d_mean.detach()), to_numpy(pred_2d_logvar.detach())
        inputs_ = to_tensor(inputs)
        for idx in range(self.network_size):
            pred_2d_mean, pred_2d_logvar = self.model_list[idx](inputs_)
            ensemble_mean[idx,:,:], ensemble_logvar[idx,:,:] \
                = to_numpy(pred_2d_mean.detach()), to_numpy(pred_2d_logvar.detach())
        means = ensemble_mean[idxs, range(ensemble_mean.shape[1]), :]   # [idxs, range(ensemble_mean.shape[1])] 좌표에 맞춰 sampling
        logvars = ensemble_logvar[idxs, range(ensemble_logvar.shape[1]), :]
        sigma = np.sqrt(np.exp(logvars))
        next_states = np.random.normal(means, sigma, size=means.shape)
        return ensemble_mean, ensemble_logvar, next_states

    def train(self, inputs, targets, batch_size=128, epochs=5):
        '''
            it is copied from a MBPO code
            <Arguments>
            inputs: (np.ndarray, [num_epi_per_iter * horizon_len, stste_dim + action_dim]):
            targets: (np.ndarray, [num_epi_per_iter * horizon_len, stste_dim]): next_state values

            <Output>
            Store indices of elite models that generate a minimal loss for the inputs and targets among ensemble models
        '''
        # Suffle the data
        rows = inputs.shape[0]
        shuffled_indices = np.random.permutation(rows)
        inputs = inputs[shuffled_indices, :]
        targets = targets[shuffled_indices, :]
        # Sample data indices for different models
        indices = [get_indices(rows) for _ in range(self.network_size)]

        for epoch in range(epochs):
            print(" Train Epoch {}/{}".format(epoch+1, epochs))
            batch_mean_losses = np.empty((1,), object)
            for start_pos in range(0, inputs.shape[0], batch_size):
                # sample a batch and get inputs and targets
                # input_ = to_tensor(inputs[start_pos : start_pos + batch_size])
                # target_ = to_tensor(targets[start_pos : start_pos + batch_size])
                input_ = [to_tensor(inputs[indices[x][start_pos : start_pos + batch_size]]) for x in range(self.network_size)]
                target_ = [to_tensor(targets[indices[x][start_pos : start_pos + batch_size]]) for x in range(self.network_size)]
                losses = np.empty((1,), object)
                for idx in range(self.network_size):
                    # train model via maximum likelihood - refer to the paper
                    mean, var = self.model_list[idx](input_[idx])
                    loss = self.model_list[idx].loss(mean, var, target_[idx])
                    if idx == 0:
                        # print('model[0] loss:', loss)
                        pass
                    self.model_list[idx].model_train(loss)
                    # compute mean_losses
                    loss = loss.detach().cpu().numpy()
                    loss = np.array([loss])
                    losses = np.concatenate((losses, loss), axis=0) if losses.item(-1) != None else loss
                mean_losses = np.mean(losses)
                mean_losses = np.array([mean_losses])
                batch_mean_losses = np.concatenate((batch_mean_losses, mean_losses), axis=0) if batch_mean_losses.item(-1) != None else mean_losses
                # print("mean_losses:", mean_losses)
            mean_batch_mean_losses = np.mean(batch_mean_losses)
            print("mean_batch_mean_losses:", mean_batch_mean_losses)


    def save_checkpoint(self, path):
        '''
            save data: {
                'network_size': scalar 
                'models_state_dict': list [model1_state_dict(), model2_state_dict(), ...],
                'optimizers_state_dict': list [optimizer1_state_dict(), optimizer2_state_dict(), ...],
            }
        '''
        models_checkpoint = []
        optimizers_checkpoint = []
        for model in self.model_list:
            model_state_dict = model.state_dict()
            optimizer_state_dict = model.optimizer.state_dict()
            models_checkpoint.append(model_state_dict)
            optimizers_checkpoint.append(optimizer_state_dict)
        
        torch.save({
            'network_size': self.network_size,
            'models_state_dict': models_checkpoint,
            'optimizers_state_dict': optimizers_checkpoint
        }, path)
    
    def load_checkpoint(self, path, eval=False):
        checkpoint = torch.load(path)
        assert self.network_size == checkpoint['network_size'], "network size should be equivalent"
        for idx in range(self.network_size):
            self.model_list[idx].load_state_dict(checkpoint['models_state_dict'][idx])
            self.model_list[idx].optimizer.load_state_dict(checkpoint['optimizers_state_dict'][idx])

            if eval == False:
                self.model_list[idx].train()
            else:
                self.model_list[idx].eval()
