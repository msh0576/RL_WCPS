import torch
import random
import numpy as np
from Dreamer_project.env import CONTROL_SUITE_ENVS, ControlSuiteEnv, Env, GYM_ENVS, EnvBatcher, MY_ENVS


class TOTAL_ENV():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, num_plant=1):
        if env not in CONTROL_SUITE_ENVS:
            raise Exception('env {} is not allowable'.format(env))
        
        # self._env = ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)    # original
        self.num_plant = num_plant
        self.schedule_size = num_plant + 1
        self._env_list = []
        for plant_id in range(num_plant):
            tmp_env = ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, camera_id=plant_id)
            # env.max_episode_steps = 500
            self._env_list.append(tmp_env)

    def reset(self):
        '''
            Returns:
                output state: tensor(CPU), [1, observation_size]
                initialize prev_commands: list len: # of plant: [tensor:[command_size, ], tensor: ..., ]
        '''
        observation = torch.cat([self._env_list[idx].reset() for idx in range(len(self._env_list))], dim=1)
        self.prev_commands = [torch.zeros(self._env_list[0].action_size) for _ in range(len(self._env_list))]
        return observation
    
    def step(self, action, action_repeat_render=False):
        '''
            Args:
                action: tensor(GPU-training, CPU-test), [1, size (=schedule, control commands) ]
                        where the schedule is a softmax distribution. We need to make a decision of the schedule
            Returns:
                observation: tensor(GPU or CPU), [1, observation_size]
                reward: scalar
                done: bool
        '''
        # print("action:", action)
        schedule_dist, commands = action[0][:self.schedule_size], action[0][self.schedule_size:]
        schedule = torch.argmax(schedule_dist)
        # print("schedule:", schedule)

        observations = []
        envs_rewards = 0
        envs_done = False
        for key, env in enumerate(self._env_list):
            '''
            scheduled system applies their command, and others maintains its previous action
            '''
            if int(schedule.item()) == key:
                self.prev_commands[key] = commands[key].view(1)
            # self.prev_commands[key]: tensor(GPU), [1, ]
            observation, reward, done = env.step(self.prev_commands[key].cpu(), action_repeat_render=action_repeat_render) 

            observations.append(observation)
            envs_rewards += reward
            if (done == True and envs_done == False):
                envs_done = True
        # print("schedule:{}, commands:{}".format(schedule.item(), self.prev_commands))
        info = {'schedule': schedule.item()}
        observations = torch.cat([observations[idx] for idx in range(len(observations))], dim=1)
        # print("observations:", observations.shape)    # tensor(GPU or CPU), [1, observation_size]
        return observations, envs_rewards, envs_done, info


    def render(self):
        for env in self._env_list:
            env.render()

    def close(self):
        for env in self._env_list:
            env.close()

    @property
    def observation_size(self):
        return self._env_list[0].observation_size * self.num_plant
    
    @property
    def action_size(self):
        '''
        Return:
            size: command size of all systems + schedule size
        '''
        return (self._env_list[0].action_size * self.num_plant) + self.schedule_size
        # return (self._env_list[0].action_size * self.num_plant)
    
    def sample_random_action(self):
        '''
            Return:
                action: tensor(GPU), schedule + control command, [1, schedule_size + action_size]
        '''
        command = torch.cat([self._env_list[idx].sample_random_action() for idx in range(len(self._env_list))])

        schedule_dist = torch.rand(self.schedule_size, dtype=torch.float32)
        return torch.cat((schedule_dist, command), dim=0).unsqueeze(0)
        # return command.unsqueeze(0)
    
    def _max_episode_steps(self):
        return self._env_list[0].max_episode_steps
