from Util.utils import to_tensor
import Plant.pendulumParam as P
from itertools import count
import numpy as np
from Models.model import PENN


class agent_PETS:
    '''
        Agent directly interacts with Environment
    '''
    def __init__(self, args, env, device, path):
        '''
            <Arguments>
            model: neural network model, where it is PENN (probabilistic ensemble neural network)
        '''
        self.args = args
        self.env = env
        self.device = device
        self.path = path
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']

    def sample(self, policy, total_step=0, t_start=0.):
        '''
            <Arguments>
                policy: MPC controller
            <Return>
                info = {
                    'states': np.ndarray :
                    'actions': np.ndarray :
                    'next_states': np.ndarray :
                    'rewards': np.ndarray :
                    'dones': np.ndarray :
                    'sum_reward': scalar : 
                    'total_step': scalar : total_step
                }
        '''
        states, actions, next_states, rewards, dones, sum_reward = [], [], [], [], [], 0

        policy.reset()

        # samples 
        state = self.env_reset()
        checkpoint_cnt = 0
        for t in count(start=t_start, step=P.Ts):
            if t >= P.t_end:
                break
            
            if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                checkpoint_cnt += 1
                total_step += 1
                # print("time step: {}/{}".format(round(t,3), P.t_end))
                action = policy.act(state, t)   # np.ndarray, [1, ]
                next_state, reward, done = self.env.step(action.item(), t)
                # print("state:", state)
                # print("reward:", reward)

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                sum_reward += reward

                state = next_state
                if done:
                    break
            else:
                self.env.update_plant_state(t) # plant status update
        info = {
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'sum_reward': sum_reward,
            'total_step': total_step 
            }
        return info


    def env_reset(self):
        self.current_state = self.env.reset()
        return self.current_state

