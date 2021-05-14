import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import AOP_project.utils

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Github: https://github.com/sfujim/TD3

# We (AOP) use mostly same hyperparameters as original

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hs=(400,300)):
        super(Actor, self).__init__()

        self.num_hidden = len(hs)

        self.l1 = nn.Linear(state_dim, hs[0])
        self.l2 = nn.Linear(hs[0], hs[1])

        if self.num_hidden == 2:
            self.l3 = nn.Linear(hs[1], action_dim)
        else:
            self.l3 = nn.Linear(hs[1], hs[2])
            self.l4 = nn.Linear(hs[2], action_dim)
        
        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        if self.num_hidden == 2:
            x = self.max_action * torch.tanh(self.l3(x)) 
        else:
            x = F.relu(self.l3(x))
            x = self.max_action * torch.tanh(self.l4(x))
        
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hs=(400,300)):
        super(Critic, self).__init__()

        self.num_hidden = len(hs)

        # Q1 architecture
        self.l1_1 = nn.Linear(state_dim + action_dim, hs[0])
        self.l2_1 = nn.Linear(hs[0], hs[1])

        if self.num_hidden == 2:
            self.l3_1 = nn.Linear(hs[1], 1)
        else:
            self.l3_1 = nn.Linear(hs[1], hs[2])
            self.l4_1 = nn.Linear(hs[2], 1)

        # Q2 architecture
        self.l1_2 = nn.Linear(state_dim + action_dim, hs[0])
        self.l2_2 = nn.Linear(hs[0], hs[1])

        if self.num_hidden == 2:
            self.l3_2 = nn.Linear(hs[1], 1)
        else:
            self.l3_2 = nn.Linear(hs[1], hs[2])
            self.l4_2 = nn.Linear(hs[2], 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1_1(xu))
        x1 = F.relu(self.l2_1(x1))

        if self.num_hidden == 2:
            x1 = self.l3_1(x1)
        else:
            x1 = F.relu(self.l3_1(x1))
            x1 = self.l4_1(x1)

        x2 = F.relu(self.l1_2(xu))
        x2 = F.relu(self.l2_2(x2))

        if self.num_hidden == 2:
            x2 = self.l3_2(x2)
        else:
            x2 = F.relu(self.l3_2(x2))
            x2 = self.l4_2(x2)

        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1_1(xu))
        x1 = F.relu(self.l2_1(x1))

        if self.num_hidden == 2:
            x1 = self.l3_1(x1)
        else:
            x1 = F.relu(self.l3_1(x1))
            x1 = self.l4_1(x1)

        return x1


class TD3():

    def __init__(self, state_dim, action_dim, max_action, hs=(400,300), device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.actor = Actor(state_dim, action_dim, max_action, hs).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, hs).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, 
        batch_size=100, discount=0.99, 
        tau=0.005, policy_noise=0.2, 
        noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
