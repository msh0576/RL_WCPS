
import numpy as np
import pickle, torch
import Plant.pendulumParam as P
from Util.utils import to_tensor, set_cuda
from Environments.Wire_Environment import wire_environment
from Models.sample_env import EnvSampler
from Models.DDPG import DDPG
from Common.replay_memory import ReplayMemory
from itertools import count
from Models.model_MBPO import Ensemble_Model
from Models.predict_env import PredictEnv

path = '/home/sihoon/works/RL_WCPS-master/DeepWNCS/Inverted_Pendulum_Simulation'

MBPO_Log_path = path + '/LogData/log/MBPO_log_.pickle'
MBPO_Model_path = path + '/LogData/model/MBPO_model_.pickle'


class MultipendulumSim_MBPO():
    def __init__(self, conf, env_sampler, predict_env, env_pool, model_pool, cuda_info):
        self.args = conf
        self.is_cuda, self.device = cuda_info
        self.agent = DDPG(conf, self.device)
        self.env_sampler = env_sampler
        self.predict_env = predict_env
        self.env_pool = env_pool
        self.model_pool = model_pool


    def train(self, model_path, cont=False):
        if cont == True:
            self.agent.Load_checkpoint(model_path, eval=False)
            epi_start, total_step = self.agent.Load_info(model_path)
            print("epi_start, total_step:", epi_start, total_step)
        else:
            epi_start = 0
            total_step = 0
        
        self.exploration_before_start(self.env_pool, self.agent)
        reward_sum = 0
        rollout_length = 1
        

        for epi_step in range(epi_start, self.args['num_episode']):
            start_step = total_step
            train_policy_steps = 0
            print("--- episode %s ---"%(epi_step))
            for t in count(start=P.t_start, step=P.Ts):
                if t >= P.t_end:
                    break

                cur_step = total_step - start_step
                
                if cur_step > 0 and cur_step % self.args['model_train_freq'] == 0:
                    # epi done 이 발생했을 때 current time 이 reset 되는 형태가 아닌듯...
                    print("current time:", t)
                    self.train_predict_model(self.env_pool, self.predict_env)
                    new_rollout_length = self.set_rollout_length(self.args, epi_step)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        # self.model_pool = resize_model_pool(self.args, rollout_length, self.model_pool)
                    
                    self.rollout_model(self.env_pool, self.agent, self.predict_env, self.model_pool, rollout_length)
                
                cur_state, action, next_state, reward, done = self.env_sampler.sample(self.agent, t)
                self.env_pool.push(cur_state, action[0], reward, next_state, done)
                # print("env_pool size: %s, min_pool_size:%s"%(len(self.env_pool), self.args['min_pool_size']))
                if len(self.env_pool) > self.args['min_pool_size']:
                    train_policy_steps += self.train_policy_repeats(total_step, train_policy_steps, cur_step, self.env_pool, self.model_pool, self.agent)
                

                total_step += 1
                if total_step % 1000 == 0:
                    '''
                    <문제>: evaluate를 할때 현재 환경 모델의 상태를 reset하고 끝날때까지 실행시킨다. 
                    끝난 후, reset 된 환경 상태부터 학습을 계속하기 때문에, 학습도중에 상태가 다시 reset 되어 버리는 문제 발생.
                    꼭 1000번마다 한번씩 evaluate_model을 할 필요가 있는건가???
                    만약 필요하다면 새로운 env_model을 만들어놓고 현재 파라미터만 넣어서 평가하는식으로 해야 할 듯...
                    '''
                    print("Save checkpoint - epi_step, total_step:", epi_step, total_step)
                    curr_info = {
                            'epi_step': epi_step,
                            'total_step': total_step
                            }
                    self.agent.Save_checkpoint(model_path, curr_info)
                    # sum_reward = self.evaluate_model()
                    # logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                    # print('total_step: %s, sum_reward: %s'%(total_step, sum_reward))
                if done == True:
                    print("epi done at ",t)
                    break

    def evaluate_model(self):
        # print("<evaluate_model start>")
        self.env_sampler.current_state = None
        sum_reward = 0
        done = False
        for t in count(start=P.t_start, step=P.Ts):
            if t >= P.t_end:
                break
            _, _, _, reward, done = self.env_sampler.sample(self.agent, t)
            sum_reward += reward
            if done:
                break
        # print("<evaluate_model end>")
        return sum_reward


    def exploration_before_start(self, env_pool, agent):
        '''
        every pool.push gets its shape of state [state_dim, ], action [1, ], reward [1, ], done [scalar]
        '''
        t = P.t_start
        step = 0
        while step <= self.args['init_exploration_steps']:
            for t in count(start=P.t_start, step=P.Ts):
                step += 1
                if t >= P.t_end or step > self.args['init_exploration_steps']:
                    break
                cur_state, action, next_state, reward, done = self.env_sampler.sample(agent, t)
                env_pool.push(cur_state, action[0], reward, next_state, done)
                if done:
                    break

    def train_predict_model(self, env_pool, predict_env):
        '''
            state   : [memory_len, state_dim], ndarray
            action  : [memory_len, 1], ndarray
            reward  : [memory_len, 1], ndarray
            done    : [memory_len, ]
            predict_env.model.train() -> there are ensemble env models, and parameters of them are updated

        '''
        # Get all samples from environment
        state, action, reward, next_state, done = env_pool.sample(len(env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((reward, delta_state), axis=-1)
        predict_env.model.train(inputs, labels, batch_size = 256)

    def set_rollout_length(self, args, epoch_step):
        rollout_length = (min(max(args['rollout_min_length'] + (epoch_step - args['rollout_min_epoch'])
            / (args['rollout_max_epoch'] - args['rollout_min_epoch']) * (args['rollout_max_length'] - args['rollout_min_length']),
            args['rollout_min_length']), args['rollout_max_length']))
        return int(rollout_length)

    def rollout_model(self, env_pool, agent, predict_env, model_pool, rollout_length):
        '''
            env_pool 중에서 batch_size만금 샘플.
            예측 환경 모델(pred_env)에 샘플을 인풋으로해서 rollout_length 미래 구간동안 step 진행.
            여기서 elite 모델에 대해서만 rollout 진행한 output을 model_pool에 저장함.
        '''
        # print("<rollout_model start>")
        state, action, reward, next_state, done = env_pool.sample_all_batch(self.args['rollout_batch_size'])
        for i in range(rollout_length):
            # TODO: Get a batch of actions
            state_ts = to_tensor(state)
            action = agent.select_action(state_ts)  # [batch_size, 1], numpy
            next_states, rewards, terminals, info = predict_env.step(state, action)
            # TODO: Push a batch of samples
            model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
            nonterm_mask = ~terminals.squeeze(-1)   # [batch_size, 1] -> [batch_size, ]
            if nonterm_mask.sum() == 0: # 모든 batch의 next_state가 termination일 경우
                break
            state = next_states[nonterm_mask]
        # print("<rollout_model end>")

    def train_policy_repeats(self, total_step, train_step, cur_step, env_pool, model_pool, agent):
        '''
            policy network parameter updates
        '''
        if total_step % self.args['train_every_n_steps'] > 0:
            return 0
        if train_step > self.args['max_train_repeat_per_step'] * total_step:
            return 0
        for i in range(self.args['num_train_repeat']):
            env_batch_size = int(self.args['policy_train_batch_size'] * self.args['real_ratio'])    # env_batch_size = 12
            model_batch_size = self.args['policy_train_batch_size'] - env_batch_size                # model_batch_size = 244

            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

            if model_batch_size > 0 and len(model_pool) > 0:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                    np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                    np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)
            self.agent.update_parameters_for_MBPO((batch_state, batch_action, batch_reward, batch_next_state, batch_done), self.args['policy_train_batch_size'], i)

        return self.args['num_train_repeat']

def readArgs(num_plant):
    conf = {
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 1,
        "num_episode": 50,
        "memory_capacity": 10000,
        "batch_size": 100,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10,
        "update_iteration": 200,
        "init_exploration_steps": 5000,
        'model_train_freq': 250,
        "num_networks": 7,
        "num_elites": 5,
        "rollout_batch_size": 100000,
        "rollout_min_length": 1,
        "rollout_max_length": 15,
        "rollout_min_epoch": 20,
        "rollout_max_epoch": 150,
        "epoch_length": 10000,   # episode length
        "model_retain_epochs": 1,
        "min_pool_size": 1000,
        "train_every_n_steps": 1,
        "num_train_repeat": 20,
        "max_train_repeat_per_step": 5,
        "policy_train_batch_size": 256,
        "real_ratio": 0.05
        }
    pend_conf = {}
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.01, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [10, 10, 10, 10, 10]  # ms
    for i in range(num_plant):
        pend_conf['pend_%s'%(i)] = {'id': i,
                                    'amplitude': amplitude_list[i],
                                    'frequency': frequency_list[i],
                                    'trigger_time': trigger_list[i]}
    return conf, pend_conf

def main():
    # =======================================================
    # configuration
    # =======================================================
    num_plant = 1
    args, pend_conf = readArgs(num_plant)
    is_cuda, device = set_cuda()
    cuda_info = (is_cuda, device)

    # initial environment
    env = wire_environment('wire', pend_conf['pend_%s'%(0)])
    env_sampler = EnvSampler(env)

    # initial ensemble model
    env_model = Ensemble_Model(args['num_networks'], args['num_elites'], \
                                args['state_dim'], args['action_dim'], device, reward_size=1, hidden_size=200)

    # Predict environments
    predict_env = PredictEnv(env_model)

    # initial pool for env
    env_pool = ReplayMemory(args['memory_capacity'])
    # initial pool for model
    rollouts_per_epoch = args['rollout_batch_size'] * args['epoch_length'] / args['model_train_freq']
    model_steps_per_epoch = int(1*rollouts_per_epoch)   # 4000000
    new_pool_size = args['model_retain_epochs'] * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    multiPendulumSim_MBPO = MultipendulumSim_MBPO(args, env_sampler, predict_env, \
                                                    env_pool, model_pool, cuda_info)
    multiPendulumSim_MBPO.train(MBPO_Model_path, cont=False)


if __name__ == '__main__':
    main()
    # A = np.arange(24).reshape(2,3,4)
    # print("A:", A)
    # b = np.arange(0, 4)
    # print("A shape:", A[[0,1,0], [0,1,2]].shape)
    # print("A :", A[[0,1,0], [0,1,2]])
    # print("A[-1]:", A[-1])
