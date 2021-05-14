import argparse
from itertools import count

import os, sys, random
import numpy as np
from Wire_Environment import wire_environment
import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import Plant.pendulumParam as P
from Plotter.trainDataPlotter import trainDataPlotter_DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=3000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

path = '/home/sihoon/works/RL_WCPS-master/DeepWNCS/Inverted_Pendulum_Simulation'
directory = path + '/LogData/log/'
Tmp_DDPG_Log_ = path + '/LogData/log/Tmp_DDPG_log_.pickle'

num_plant = 1
configuration = {
    'num_plant' : num_plant,
    'state_dim' : 5*num_plant,
    'action_dim': 1,
    "num_episode": 600,
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
amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
frequency_list = [0.01, 0.15, 0.2, 0.2, 0.2]
trigger_list = [10, 10, 10, 10, 10]  # ms
for i in range(num_plant):
    pend_configuration['pend_%s'%(i)] = {'id': i,
                                         'amplitude': amplitude_list[i],
                                         'frequency': frequency_list[i],
                                         'trigger_time': trigger_list[i]}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = wire_environment('wire', pend_configuration['pend_%s'%(0)])


state_dim = 5
action_dim = 1
max_action = 0.5
min_Val = torch.tensor(1e-7).float().to(device) # min value


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device).squeeze() # (100, 5)
            action = torch.FloatTensor(u).to(device)    # (100, 1)
            next_state = torch.FloatTensor(y).to(device).squeeze()    # (100, 5)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)    # (100, 1)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        return critic_loss.item(), actor_loss.item()

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def save_log(log_path):
    combined_stats = dict()
    combined_stats['episode/reward'] = epi_rewards
    # combined_stats['step/count'] = self.step_count
    combined_stats['episode/duration'] = epi_durations
    combined_stats['step/actor_loss'] = actor_losses
    combined_stats['step/critic_loss'] = critic_losses
    combined_stats['episode/count'] = episode
    with open(log_path, 'wb') as f:
        pickle.dump(combined_stats,f)
def load_log(log_path):
    with open(log_path, 'rb') as f:
        data = pickle.load(f)
    return data

def debug_grad(network, layer_name):
    for name, param in network.named_parameters():
        if name == layer_name:
            print("%s // grad:%s"%(name, param.grad))

epi_rewards, epi_durations, episode = [], [], []
actor_losses, critic_losses = [], []

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                # env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0

        for i in range(args.max_episode):
            total_reward = 0
            step =0
            epi_duaration = 0.
            state = env.reset()
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 1 ms, schedule udpate
                    action = agent.select_action(state)
                    action = (action + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(-0.5, 0.5)
                    # print("action:", action.item())
                    next_state, reward, done = env.step(action.item(), t)
                    # if args.render and i >= args.render_interval : env.render()
                    agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                    state = next_state
                    step += 1
                    total_reward += reward
                    if done:
                        break
                else:   # every 1 ms
                    env.update_plant_state(t) # plant status update
                t = t + P.Ts
            total_step += step+1
            print("Episode: \t{} Total Reward: \t{:0.2f}  Duration: \t{:0.3f}".format(i, total_reward, epi_duration))
            critic_loss, actor_loss = agent.update()

            # episode end
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            episode.append(i)
            epi_durations.append(epi_duration)
            epi_rewards.append(total_reward)
            if i % args.log_interval == 0:
                agent.save()
                save_log(Tmp_DDPG_Log_)
    else:
        raise NameError("mode wrong!!!")

def plot_cumulate_reward(log_path):
    '''
    plot log data on training results
    '''
    log_data = load_log(log_path)
    log_data_plotter = trainDataPlotter_DDPG()
    log_data_plotter.plot(log_data)

if __name__ == '__main__':

    # main()
    plot_cumulate_reward(Tmp_DDPG_Log_)
