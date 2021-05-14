import random
import numpy as np
import torch

class ReplayMemory:
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
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        # print("<sample>")
        # print("action:", action.shape)
        # print("state:", state.shape)
        # print("reward:", reward.shape)
        # print("done:", done.shape)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
