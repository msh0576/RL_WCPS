# from collections import deque
import numpy as np
import random

import pickle as pickle

# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
class rpm(object):
    #replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

        import threading
        self.lock = threading.Lock()
        
    def append(self, obj):
        self.lock.acquire()
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size()-self.buffer_size):]

        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size

        else:
            self.buffer.append(obj)

        self.lock.release()

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''

        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            k = np.stack((item[i] for item in batch),axis=0)
            if len(k.shape)==1: k.shape+=(1,)
            res.append(k)

        [state_batch, action_batch, reward_batch, \
         next_state_batch, terminal_batch] = res
        
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
    
    def save(self, pathname):
        self.lock.acquire()
        pathname = '{}/memory.pkl'.format(pathname)
        pickle.dump([self.buffer,self.index], open(pathname, 'wb'), True)
        print('memory dumped into', pathname)
        self.lock.release()
        
    def load(self, pathname):
        pathname = '{}/memory.pkl'.format(pathname)
        [self.buffer,self.index] = pickle.load(open(pathname, 'rb'))
        print('memory loaded from', pathname)
