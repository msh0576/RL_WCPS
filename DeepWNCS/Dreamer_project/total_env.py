import torch
import random
import numpy as np
from Dreamer_project.env import CONTROL_SUITE_ENVS, ControlSuiteEnv, Env, GYM_ENVS, EnvBatcher, MY_ENVS
from Dreamer_project.utils import sequentialSchedule, randomSchedule

##### By sihoon #####
class TOTAL_ENV():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, num_plant=1, schedAlgo='sequential'):
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

        if schedAlgo == 'sequential':
            self.schedule_algorithm = sequentialSchedule
        elif schedAlgo == 'random':
            self.schedule_algorithm = randomSchedule
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
        - 스케줄이 안 된 시스템은 이전 action을 그대로 사용함
        - 특정 시스템을 스케줄 했을 때만, 자신의 state 정보를 확인할 수 있음. but, replay memory에 저장하는 정보의 형태도 조정이 필요함...

            Args:
                action: tensor(GPU-training, CPU-test), [1, size (=schedule, control commands) ]
                        where the schedule is a softmax distribution. We need to make a decision of the schedule
            Returns:
                observation: tensor(GPU or CPU), [1, observation_size]
                reward: scalar
                done: bool
        '''
        # print("action:", action)
        schedule, commands = action[0][:1], action[0][1:]
        # schedule = torch.argmax(schedule_dist)
        # print("schedule:", schedule)
        # print("before commands:", commands)

        observations = []
        envs_rewards = 0
        envs_done = False
        reward_list = []
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
            reward_list.append(reward)
            if (done == True and envs_done == False):
                envs_done = True
        # print("schedule:{}, commands:{}".format(schedule.item(), self.prev_commands))
        command_actions = torch.cat(self.prev_commands, dim=0)
        info = {'schedule': schedule.item(),
                'command_actions': command_actions,
                'reward_list': reward_list}
        observations = torch.cat([observations[idx] for idx in range(len(observations))], dim=1)
        # print("observations:", observations.shape)    # tensor(GPU or CPU), [1, observation_size]
        return observations, envs_rewards, envs_done, info

    def select_actions(self, state, agents, evaluate=False):
        '''
            Inputs:
                agents: agent_list
                state: entire system states
            Outputs:
                action: torch(CPU), [1, (schedule_size + each_command_size*num_plant)]
        '''
        # specific env observation size
        spcf_obs_size = [each_env.observation_size for each_env in self._env_list]

        schedule = self.schedule_algorithm(self.num_plant + 1)

        command_actions = []
        for idx, agent in enumerate(agents):
            spcf_obs_criet = sum(spcf_obs_size[:idx]) if idx != 0 else 0
            command_actions.append(agent.select_action(state[:, spcf_obs_criet:(spcf_obs_criet+spcf_obs_size[idx])], evaluate=evaluate))  # Sample action from policy    # tensor(CPU), [1, action_size(command_action)]
        command_actions = torch.Tensor([command_actions])
        action = torch.cat([schedule.view(1,1), command_actions], dim=1)
        return action

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
            size: command size of all systems
        '''
        # return (self._env_list[0].action_size * self.num_plant) + self.schedule_size
        return (self._env_list[0].action_size * self.num_plant)
    
    def sample_random_action(self):
        '''
            Return:
                action: tensor(CPU), schedule + control command, [1, schedule_size + action_size]
        '''
        command = torch.cat([self._env_list[idx].sample_random_action() for idx in range(len(self._env_list))])

        # schedule_dist = torch.rand(self.schedule_size, dtype=torch.float32)
        schedue = self.schedule_algorithm(self.schedule_size)
        return torch.cat((schedue, command), dim=0).unsqueeze(0)
        # return command.unsqueeze(0)
    
    def _max_episode_steps(self):
        return self._env_list[0].max_episode_steps

