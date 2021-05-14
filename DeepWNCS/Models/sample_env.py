from Util.utils import to_tensor
import Plant.pendulumParam as P
from itertools import count
import numpy as np


class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env
        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, t):
        '''
            <output>
            sample an action and takes the action to the env
            action : [batch_size, 1]
        '''
        if self.current_state is None:
            print("reset() in sample_env at ", t)
            self.env_reset()

        cur_state = self.current_state
        cur_state_ts = to_tensor(cur_state).reshape(-1).unsqueeze(0)    #[1, state_dim]
        action = agent.select_action(cur_state_ts)  # [batch_size, 1]
        next_state, reward, done = self.env.step(action.item(), t)

        # self.path_length += 1
        # self.sum_reward += reward

        # if done or self.path_length >= self.max_path_length:
        if done:
            print("env done in sample_env at ", t)
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state
        return cur_state, action, next_state, reward, done

    def env_reset(self):
        self.current_state = self.env.reset()
        return self.current_state

class EnvSampler_PETS(EnvSampler):
    def __init__(self, env, agent):
        super().__init__(env)
        self.agent = agent

    def horizon_sample(self, t_start, horizon):
        '''
            <argements>
            horizon: it seems like a rollout length
            
            <output>
            info = {'states': (np array) [state-1, ..., state-horizon],
                    'actions': (np array) [action-1, ..., action-horizon],
                    'next_states': (np array) [next_state-1, ..., next_state-horizon],
                    'rewards': (np array) [reward-1, ..., reward-horizon],
                    'dones' : (np array) 
                    'sum_reward': (scalar) sum_reward
                    }
        '''

        states, actions, next_states, rewards, dones, sum_reward = [], [], [], [], [], 0
        # policy.reset() 은 어떻게 하지???

        # samples 
        horizon_cnt = 0
        for t in count(start=t_start, step=P.Ts):
            if horizon_cnt >= horizon or t >= P.t_end:
                break
            state, action, next_state, reward, done = super().sample(self.agent, t)

            states.append(state)
            actions.append(action[0])
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(np.array([done]))
            sum_reward += reward
            
            info = {'states': np.array(states),
                    'actions': np.array(actions),
                    'next_states': np.array(next_states),
                    'rewards': np.array(rewards),
                    'dones': np.array(dones),
                    'sum_reward': sum_reward}
            horizon_cnt += 1
            if done:
                break

        return info

    
    def get_plant_status(self):
        return self.env.get_current_plant_status()