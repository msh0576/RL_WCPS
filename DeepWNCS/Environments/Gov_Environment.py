import torch
import numpy as np
import random
from itertools import count

from Environments.Environment import environment
from Environments.Wire_Environment import wire_environment


class Env:
    def __init__(self, args, network='wireless'):
        self.args = args
        # default environment setting
        pend_conf = {}
        amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
        frequency_list = [0.01, 0.15, 0.2, 0.2, 0.2]
        trigger_list = [10, 10, 10, 10, 10]  # ms
        for i in range(args.num_plant):
            pend_conf['pend_%s'%(i)] = {'id': i,
                                        'amplitude': amplitude_list[i],
                                        'frequency': frequency_list[i],
                                        'trigger_time': trigger_list[i]}

        if network == 'wireless':
            # self.env = environment()
            raise NotImplementedError
        else:
            self._env = wire_environment('wire', pend_conf['pend_%s'%(0)])
        
        self.action_repeat = args.action_repeat
        self.max_episode_length = args.max_episode_length
    
    def reset(self):
        self.t = 0 # Reset internal timer
        state = self._env.reset()
        return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    
    def step(self, action):
        '''
            step을 밟을때마다 action_repeat 만큼의 reward를 계산함
        '''
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            # print("time:{}s".format(round(self.t*0.01,3)))
            next_state, reward_, done, _ = self._env.step(action.item(0), self.t*0.01)
            reward += reward_
            self.t += 1
            if done or self.t == self.max_episode_length:
                break
        observation = torch.tensor(next_state, dtype=torch.float32).unsqueeze(dim=0)    # tensor, [1, 5]

        return observation, reward, done
    @property
    def action_size(self):
        return self._env.action_spec().shape[0]
    
    @property
    def observation_size(self):
        return self._env.observation_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))
    
    