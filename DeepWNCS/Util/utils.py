
import torch, pickle
from torch.autograd import Variable
import numpy as np
import random

def set_cuda():
    is_cuda = torch.cuda.is_available()
    print("torch version: ", torch.__version__)
    print("is_cuda: ", is_cuda)
    print(torch.cuda.get_device_name(0))
    if is_cuda:
        device = torch.device("cuda:0")
        print("Program will run on *****GPU-CUDA***** ")
    else:
        device = torch.device("cpu")
        print("Program will run on *****CPU***** ")

    return is_cuda, device

def identity(x):
    return x


def entropy(p):
    return -torch.sum(p * torch.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -torch.sum(torch.exp(log_p1)*(log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot

'''
def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
'''

def to_numpy(var):
    return var.cpu().numpy()

'''
def to_tensor(x, is_cuda = True, device = "cuda:0" , dtype = np.float32):
    if is_cuda:
        tensor_var = torch.from_numpy(x).float().to(device)
    else:
        tensor_var = torch.from_numpy(x).float()
    return tensor_var
'''
def to_tensor(ndarray, is_cuda = True, device = "cuda:0", requires_grad=False):
    dtype = torch.cuda.FloatTensor if is_cuda == True else torch.FloatTensor
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype).to(device)


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def save_checkpoint_dataset(dataset, path):
        with open(path, 'wb') as f:
            pickle.dump(dataset,f)

def load_checkpoint_dataset(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

import ujson as json
import pickle
def make_dataset(dataset_path, env, agents, model_path, num_plant):
    '''
        samples:
            evals: 매 스탭마다 모든 시스템의 observation 및 action 정보 [o_t^1, o_t^2, a_t^1, a_t^2]
            values: obs sequence 중에서 현재 시점에 스케줄 된 시스템만 값을가지고, 나머지는 0. But, action 값은 모두 존재함.
                    ex. schedule_t = 0, [o_t^1, 0, a_t^1, a_t^2]
            masks: loss 된 정보는 0, 아니면 1. ex. [1, 0, 1, 1]
            eval_masks: evals에는 존재하지만, values에서는 값이 loss 가 되어 평가가 필요한 정보. ex) [0, 1, 0, 0]
    '''
    TRAJ_SAMPLE_LEN = 99

    
    each_observation_size, each_command_action_size = env.observation_size/num_plant, env.action_size/num_plant

    ## loads system models
    for idx, agent in enumerate(agents):
        agent.load_model(model_path+"_{}".format(idx), evaluate='True')

    avg_reward, episodes, t = 0., 400, 0
    test_render = False
    episodes_dataset = []
    for epi_idx  in range(episodes):
        episode_samples = {
                'forward': {},
                'backward': {},
                'label': [],
                'is_train':[]
                }
        max_traj = env._env_list[0].max_episode_length - TRAJ_SAMPLE_LEN - 1
        sample_traj = random.randint(0, max_traj)

        if epi_idx == episodes-1:
            test_render = True
            schedules = []
        state = env.reset()
        episode_reward = 0
        done = False

        
        forwards_list = []
        # last_obs_time = [0 for _ in range(env.observation_size + env.action_size)]
        delta = [0. for _ in range(env.observation_size + env.action_size)]
        while not done:
            forward_dic = { 
                        'values':[],
                        'masks': [],
                        'deltas': [],
                        'forwards': [],
                        'evals': [],
                        'eval_masks': [],
                        'times': []
                        }
            action = env.select_actions(state, agents, evaluate=True)
            next_state, reward, done, info = env.step(action, test_render)
            episode_reward += reward
            if epi_idx == episodes-1:
                schedules.append(info['schedule'])
                # metrics['schedule'].append(info['schedule'])
                # metrics['steps'].append(t)
            
            if t >= sample_traj:
                # Set samples
                eval = torch.cat([state.squeeze(0), info['command_actions']], dim=0)
                schedule = action[0][:1]
                obs_criteria = int(each_observation_size * schedule)
                action_criteria = int(env.observation_size + int(each_command_action_size * schedule))
                mask = [1 if (idx >= obs_criteria and idx < obs_criteria + each_observation_size and idx < env.observation_size) or \
                                (idx >= env.observation_size and idx < env.observation_size + env.action_size) \
                                else 0  for idx in range(len(eval))]
                values = eval * torch.Tensor(mask)
                eval_masks = [0 if mask_i == 1 else 1 for mask_i in mask]

                # Store samples
                forward_dic['values'] = [round(value, 4) for value in values.tolist()]
                forward_dic['masks'] = mask
                forward_dic['evals'] = [round(e, 4) for e in eval.tolist()]
                forward_dic['eval_masks'] = eval_masks
                if t > 0:
                    # delta = [t - each_time for each_time in last_obs_time]
                    # last_obs_time = [t if m == 1 else last_obs_time[idx] for idx, m in enumerate(mask)]
                    delta = [delta[idx] if m == 1 else delta[idx]+1 for idx, m in enumerate(mask)]
                
                forward_dic['deltas'] = delta
                forward_dic['times'] = t
                forwards_list.append(forward_dic)

                # print("------------------")
                # print("t: {}, schedule:{}".format(t, schedule))
                # print("delta:", delta)
                # print("eval:", eval)
                # print("mask:", mask)
                # print("values:", values)
                # print("eval_masks:", eval_masks)
                # print("forward_dic:", forward_dic)

            state = next_state
            t += 1

            if t >= sample_traj + TRAJ_SAMPLE_LEN:
                t = 0
                break
        episode_samples['forward'] = forwards_list
        episode_samples['backward'] = list(reversed(forwards_list))

        # revise deltas in episode_samples['backward']
        rev_delta = [0. for _ in range(env.observation_size + env.action_size)]
        for idx, backward in enumerate(episode_samples['backward']):
            if idx > 0:
                rev_delta = [rev_delta[idx] if m == 1 else rev_delta[idx]-1 for idx, m in enumerate(backward['masks'])]
            episode_samples['backward'][idx]['deltas'] = rev_delta

        episodes_dataset.append(episode_samples)
    env.close()
    # pickle file store
    with open(dataset_path, 'wb') as outfile:
        pickle.dump(episodes_dataset, outfile)
    
    # update_TEXT(dataset_path, episodes_dataset)




def update_TEXT(file_, content, mode='write'):
    '''
        INPUT:
            content: dict
    '''
    if mode == 'write':
        with open(file_, 'w') as f:
            f.write(str(content))
            # json.dump(content, f)
    elif mode == 'update':
        with open(file_, 'a') as f:
            f.write(str(content))


