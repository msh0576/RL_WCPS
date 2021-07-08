import random
import numpy as np
import torch
from BRITS_project.data_loader import to_tensor_dict, collate_fn
import pickle
import copy as cp

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        '''
            Input:
                state: torch(CPU), [1, 5]
                action: torch(CPU), [1, 1]
                reward: torch(CPU), [1,]
                next_state: torch(CPU), [1,5]
                mask: torch(CPU), [1,]
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
            Output:
                state: torch(CPU), [batch, 1, 5]
                action:  torch(CPU), [batch, 1, 1]
                reward: torch(CPU), [batch, 1]
                next_state: torch(CPU), [batch, 1, 5]
        '''
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        # print("<sample>")
        # print("action:", action.shape)
        # print("state:", state.shape)
        # print("reward:", reward.shape)
        # print("next_state:", next_state.shape)
        # print("done:", done.shape)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class RewardMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        reward, next_state = map(torch.stack, zip(*batch))

        return reward, next_state

    def __len__(self):
        return len(self.buffer)


class LatentODEMemory():
    def __init__(self, n_traj, env_obs_size, env_action_size, num_plant, seed, device):
        random.seed(seed)
        self.n_traj = n_traj
        self.env_obs_size = env_obs_size
        self.env_action_size = env_action_size
        self.num_plant = num_plant
        self.device = device

        self.epi_buff = [None]
        self.traj_buff = []
        self.epi_pos = 0
        self.traj_pos = 0
        self.each_obs_size, self.each_action_size = self.env_obs_size/num_plant, self.env_action_size/num_plant

    def push(self, obs, command_action, tp):
        '''
            Input:
                obs: tensor(CPU), [1, 5]
                command_action: tensor(CPU), [1, 1]
                tp: scalar (int): time point
        '''
        assert (obs.size()[0] == 1 and obs.size()[1] == self.each_obs_size)
        assert (command_action.size()[0] == 1 and command_action.size()[1] == self.each_action_size)
        self.traj_buff.append(None)

        observed_data = torch.cat([obs, command_action], dim=1)
        self.traj_buff[self.traj_pos] = (observed_data, torch.FloatTensor([tp]))
        self.epi_buff[self.epi_pos] = self.traj_buff

        self.traj_pos += 1

    def get_latest_sample(self, tp):
        '''
        Output:
            dataset: {
                'observed_data': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'observed_tp': tensor(GPU), [100,]
                'data_to_predict': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'tp_to_predict': tensor(GPU), [100,]
                'observed_mask': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'mask_predicted_data': None
                'mode': interp
                'labels': None
            }
        '''
        assert (tp > self.n_traj)

        dataset = {
            'observed_data': [],
            'observed_tp': [],
            'data_to_predict': [],
            'tp_to_predict': [],
            'observed_mask': [],
            'mask_predicted_data': None,
            'mode': 'interp',
            'labels': None
        }
        sample = self.epi_buff[self.epi_pos][-self.n_traj:]
        observed_data = [sample_[0] for sample_ in sample]
        observed_tp = [sample_[1] for sample_ in sample]

        observed_data = torch.cat(observed_data, dim=0)
        observed_tp = torch.cat(observed_tp, dim=0)
        # print('observed_data:', observed_data.shape)
        # print('observed_tp:', observed_tp.shape)

    def get_test_sample(self):
        dataset = {
            'observed_data': [],
            'observed_tp': [],
            'data_to_predict': [],
            'tp_to_predict': [],
            'observed_mask': [],
            'mask_predicted_data': None,
            'mode': 'interp',
            'labels': None
        }
        # observed_data 중, [:-5] 까지는 observed_data, [-5:] 는 data_to_predict로 사용
        epi_rand_indices = self.get_random_epi_indices(batch_size=1)
        traj_rand_indices = self.get_random_traj_indices(epi_rand_indices)

        observed_data_list = []    # each element is an episode trajectory
        observed_tp_list = []
        for idx_, traj_idx in enumerate(traj_rand_indices):
            # print("self.epi_buff[epi_rand_indices[idx_]]:", self.epi_buff[epi_rand_indices[idx_]][traj_idx:traj_idx+self.n_traj])
            samples = self.epi_buff[epi_rand_indices[idx_]][traj_idx:traj_idx+self.n_traj]
            observed_data = [sample[0] for sample in samples]
            # print('observed_data:', observed_data)
            observed_tp = [sample[1] for sample in samples]
            observed_data_list.append(observed_data)
            observed_tp_list.append(observed_tp)

    def generate_train_dataset(self, batch_size):
        '''
        Output:
            dataset: {
                'observed_data': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'observed_tp': tensor(GPU), [100,]
                'data_to_predict': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'tp_to_predict': tensor(GPU), [100,]
                'observed_mask': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                'mask_predicted_data': None
                'mode': interp
                'labels': None
            }
        '''
        dataset = {
            'observed_data': [],
            'observed_tp': [],
            'data_to_predict': [],
            'tp_to_predict': [],
            'observed_mask': [],
            'mask_predicted_data': None,
            'mode': 'interp',
            'labels': None
        }
        epi_rand_indices = self.get_random_epi_indices(batch_size)
        traj_rand_indices = self.get_random_traj_indices(epi_rand_indices)

        observed_data_list = []    # each element is an episode trajectory
        observed_tp_list = []
        for idx_, traj_idx in enumerate(traj_rand_indices):
            # print("self.epi_buff[epi_rand_indices[idx_]]:", self.epi_buff[epi_rand_indices[idx_]][traj_idx:traj_idx+self.n_traj])
            samples = self.epi_buff[epi_rand_indices[idx_]][traj_idx:traj_idx+self.n_traj]
            observed_data = [sample[0] for sample in samples]
            # print('observed_data:', observed_data)
            observed_tp = [sample[1] for sample in samples]
            observed_data_list.append(observed_data)
            observed_tp_list.append(observed_tp)
        
        # list to tensor
        dataset['observed_data'] = to_tensor_matrix(observed_data_list).to(self.device)
        dataset['observed_tp'] = torch.cat(observed_tp, dim=0).to(self.device)
        dataset['data_to_predict'] = to_tensor_matrix(observed_data_list).to(self.device)
        dataset['tp_to_predict'] = torch.cat(observed_tp, dim=0).to(self.device)
        dataset['observed_mask'] = torch.ones_like(dataset['data_to_predict']).to(self.device)
        # print("dataset['observed_data'] shape:",dataset['observed_data'].shape)
        # print("dataset['observed_tp']:", dataset['observed_tp'].shape)

        return dataset
        

    def set_new_buff(self):
        self.epi_pos += 1
        self.epi_buff.append(None)
        self.traj_pos = 0
        self.traj_buff = []
    
    def get_random_traj_indices(self, epi_rand_indices):
        '''
            generate random integer, and check whether a trajectory from the integer is intact
        '''
        list_ = []
        for epi_idx in epi_rand_indices:
            current_epi_buffer_len = len(self.epi_buff[epi_idx])
            while True:
                index = random.randint(0, current_epi_buffer_len)
                if index <= current_epi_buffer_len - self.n_traj:
                    break
            list_.append(index)
        return list_

    def get_random_epi_indices(self, batch_size):
        '''
            Output:
                list_: list, [batch_size,]: episode indices avilable to extract samples as n_traj
        '''
        list_ = []
        for _ in range(batch_size):
            if self.epi_pos == 0:
                epi_idx = 0
            elif self.epi_pos > 0 and self.traj_pos <= self.n_traj:
                epi_idx = random.randint(0, self.epi_pos-1)
            else:
                epi_idx = random.randint(0, self.epi_pos)
            list_.append(epi_idx)
        return list_

def to_tensor_matrix(samples):
    '''
    Input:
        samples: list, len(batch), [element1, ...]: element1 = len(n_traj) [tensor[1xsth_size], ...]
    Output:
        samples_matrix: tensor, [batch, n_traj, sth_size]
    '''
    tmp = list(map(lambda x: torch.cat(x, dim=0).unsqueeze(0), samples)) # list, [tensor(1, 100, sth_size), ...]
    samples_matrix = torch.cat(tmp, dim=0)   # tensor, [50(batch), 100, sth_size]

    return samples_matrix

class TrajectoryMemory():
    def __init__(self, capacity, traj_seq_len, env_obs_size, env_action_size, num_plant, seed):
        random.seed(seed)
        self.capacity = capacity
        self.epi_capacity = 1000
        self.traj_seq_len = traj_seq_len
        self.epi_buffer = [None]
        self.buffer = []
        self.position = 0
        self.epi_position = 0
        self.env_obs_size = env_obs_size
        self.env_action_size = env_action_size
        self.each_obs_size, self.each_action_size = self.env_obs_size/num_plant, self.env_action_size/num_plant
        
        self.delta = [0. for _ in range(env_obs_size + env_action_size)]

        episode_samples = {
                'forward': {},
                'backward': {},
                'label': [],
                'is_train':[]
                }
    
    def push_sample(self, state, schedule, command_action):
        '''
            Inputs:
                state:[o_t^1, o_t^2, ...] = [1, env_obs_size]
            Algorithm:
                miseed observations are 0, we can measure it with respect to the schedule
                values: torch, [env_obs_size + env_action_size] = [o_t^1, 0, ..., a_t^1, a_t^2, ...] when schedule = 0

        '''
        eval = torch.cat([state.squeeze(0), command_action], dim=0)
        obs_criteria = int(self.each_obs_size * schedule)
        mask = [1 if (idx >= obs_criteria and idx < obs_criteria + self.each_obs_size and idx < self.env_obs_size) or \
                        (idx >= self.env_obs_size and idx < self.env_obs_size +self.env_action_size) \
                        else 0  for idx in range(len(eval))]
        values = eval * torch.Tensor(mask)
        values = [round(value, 4) for value in values.tolist()]
        evals = [round(e, 4) for e in eval.tolist()]
        eval_masks = [0 if mask_i == 1 else 1 for mask_i in mask]
        # if len(self.buffer) > 0:
            # self.delta = [self.delta[idx] if m == 1 else self.delta[idx]+1 for idx, m in enumerate(mask)]
            # self.delta = [self.delta[idx] if m == 1 else 0 for idx, m in enumerate(mask)]
        # print("self.delta:", self.delta)

        self.push(values, mask, evals, eval_masks, self.delta)  # for brits model update
        # self.push_traj(values) # for brits model test

    def push(self, values, masks, evals, eval_masks, deltas):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # print("len(self.buffer):", len(self.buffer))      
        forward_dic = { 
                        'values':[],
                        'masks': [],
                        'deltas': [],
                        'forwards': [],
                        'evals': [],
                        'eval_masks': [],
                        'times': []
                        }
        forward_dic['values'] = values
        forward_dic['masks'] = masks
        forward_dic['evals'] = evals
        forward_dic['eval_masks'] = eval_masks
        forward_dic['deltas'] = deltas
        forward_dic['times'] = 0
        # print("forward_dic:", forward_dic)

        self.buffer[self.position] = cp.deepcopy(forward_dic)
        self.epi_buffer[self.epi_position] = cp.deepcopy(self.buffer)
        self.position += 1
        # self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size=32, repeat=1):
        '''
            Algo:
                If the batch_size == 1, then we use this function for estimation of missed obs
                Otherwise, for training the brits model
            Output:
                forward: tensor, [batch, traj_seq_len, state_size(env_obs_size + env_action_size)]
        '''
        if batch_size == 1:
            # batch_buffer = self.get_latest_samples(repeat=repeat)
            # forward = to_tensor_dict(batch_buffer['forward'])
            dataset = self.get_latest_samples()
            ret_dict = collate_fn(dataset)
            forward = ret_dict['forward']
        else:
            dataset = self.make_batch(batch_size=batch_size)
            ret_dict = collate_fn(dataset)
            forward = ret_dict['forward']

        # print("batch_size:{} -- forward:{}".format(batch_size,forward['values'].shape))
        
        ret_dict = {'forward': forward, 'backward': []}
        ret_dict['labels'] = torch.FloatTensor([0]*batch_size)
        ret_dict['is_train'] = torch.FloatTensor([0]*batch_size)
        # print("ret_dict:", ret_dict)
        return ret_dict


    def make_batch(self, batch_size):
        '''
            Algo:
                extract batch trajectories in a self.buffer
            Output:
                list, [batch_size,]: [dict, dict, ...]
        '''
        
        # print("epi_buffer len:{}, buffer len:{}".format(len(self.epi_buffer), len(self.buffer)))

        epi_rand_indices = []
        rand_indices = []
        epi_rand_indices = self.generate_epi_random_indices(batch_size)
        rand_indices = self.generate_random_indices(epi_rand_indices)
        # if self.epi_position == 1:
        #     print("rand_indices:", rand_indices)
        dataset = []
        for idx, rand_index in enumerate(rand_indices):
            batch_buffer = {'forward': [],
                        'backward': [],
                        'label': [],
                        'is_train':[]}
            traj = self.epi_buffer[epi_rand_indices[idx]][rand_index:rand_index+self.traj_seq_len]
            # if self.epi_position == 1:
            #     print("-------------")
            #     print("self.epi_buffer len:", len(self.epi_buffer[epi_rand_indices[idx]]))
            #     print("epi_rand_indices:{}, rand_index:{}".format(epi_rand_indices[idx], rand_index))
            #     print("traj:", len(traj))
            #     # print("traj:", traj)
            self.set_deltas(traj)
            backward_traj = list(reversed(traj))

            rev_delta = [0. for _ in range(self.env_obs_size + self.env_action_size)]
            for idx_, backward in enumerate(backward_traj):
                if idx_ > 0:
                    rev_delta = [rev_delta[m_idx] if m == 1 else rev_delta[m_idx]-1 for m_idx, m in enumerate(backward['masks'])]
                backward_traj[idx_]['deltas'] = rev_delta

            batch_buffer['forward'] = traj
            batch_buffer['backward'] = backward_traj
            dataset.append(batch_buffer)
        return dataset
    
    def get_latest_samples(self):
        epi_idx = self.epi_position if len(self.epi_buffer[self.epi_position]) >= self.traj_seq_len else self.epi_position-1
        traj_idx = len(self.epi_buffer[epi_idx]) - self.traj_seq_len
        dataset = []

        batch_buffer = {'forward': [],
                        'backward': [],
                        'label': [],
                        'is_train':[]}
        traj = self.epi_buffer[epi_idx][traj_idx:traj_idx+self.traj_seq_len]
        self.set_deltas(traj)
        backward_traj = list(reversed(traj))
        rev_delta = [0. for _ in range(self.env_obs_size + self.env_action_size)]
        for idx_, backward in enumerate(backward_traj):
            if idx_ > 0:
                rev_delta = [rev_delta[m_idx] if m == 1 else rev_delta[m_idx]-1 for m_idx, m in enumerate(backward['masks'])]
            backward_traj[idx_]['deltas'] = rev_delta
        batch_buffer['forward'] = traj
        batch_buffer['backward'] = backward_traj

        dataset.append(batch_buffer)
        return dataset


    def get_latest_samples_(self, repeat):
        '''
            Algo:
                get a latest samples with respect to the self.position
            Output:
                traj: list, [1(dict)]
        '''
        # test code #
        epi_rand_idx =  self.generate_epi_random_indices(1)
        traj_rand_idx = self.generate_random_indices(epi_rand_idx)
        ###
        batch_buffer = {'forward': [],
                        'backward': [],
                        'label': [],
                        'is_train':[]}
        assert self.position >= self.traj_seq_len, "[get_latest_samples()] Not enough data in a buffer"
        if self.position >= self.traj_seq_len:
            for _ in range(repeat):
                # batch_buffer['forward'].append(self.epi_buffer[self.epi_position][self.position - self.traj_seq_len:self.position])
                batch_buffer['forward'].append(self.epi_buffer[epi_rand_idx[0]][traj_rand_idx[0]:traj_rand_idx[0] + self.traj_seq_len])

        # assert len(batch_buffer['forward']) == 1, '[get_latest_samples()] batch_buffer size Error'
        for idx in range(len(batch_buffer['forward'])):
            self.set_deltas(batch_buffer['forward'][idx])

        return batch_buffer
    
    def get_empty_buffer(self):
        evals = [0 for _ in range(self.env_obs_size+self.env_action_size)]
        masks = [1 for _ in range(self.env_obs_size+self.env_action_size)]
        values = [0 for _ in range(self.env_obs_size+self.env_action_size)]
        eval_masks = [0 for _ in range(self.env_obs_size+self.env_action_size)]
        deltas = [1 for _ in range(self.env_obs_size+self.env_action_size)]
        deltas[0] = 0

        forward_dic = { 
                        'values':[],
                        'masks': [],
                        'deltas': [],
                        'forwards': [],
                        'evals': [],
                        'eval_masks': [],
                        'times': []
                        }
        forward_dic['values'] = values
        forward_dic['masks'] = masks
        forward_dic['evals'] = evals
        forward_dic['eval_masks'] = eval_masks
        forward_dic['deltas'] = deltas
        forward_dic['times'] = 0

        return forward_dic

    def set_deltas(self, one_traj_buffer):
        ## revise the deltas info
        for idx, forward_ts in enumerate(one_traj_buffer):
            if idx == 0:
                forward_ts['deltas'] = [0 for _ in range(len(forward_ts['masks']))]
            else:
                forward_ts['deltas'] = [1 + prev_delta[idx] if int(prev_mask) == 0 else 1 for idx, prev_mask in enumerate(prev_masks)]
            prev_masks = forward_ts['masks']
            prev_delta = forward_ts['deltas']

    def store_dataset(self, batch_size, dataset_path):
        dataset = self.make_batch(batch_size=batch_size)
        print("dataset store!")
        # pickle file store
        with open(dataset_path, 'wb') as outfile:
            pickle.dump(dataset, outfile)

    def new_epi_buffer(self):
        self.epi_position = (self.epi_position + 1) % self.epi_capacity
        if len(self.epi_buffer) < self.epi_capacity:
            self.epi_buffer.append(None)
        self.position = 0
        self.buffer = []
        # print("self.epi_position:{}, self.position:{}".format(self.epi_position, self.position))


    def generate_random_indices(self, epi_rand_indices):
        '''
            generate random integer, and check whether a trajectory from the integer is intact
        '''
        list_ = []
        for epi_idx in epi_rand_indices:
            current_epi_buffer_len = len(self.epi_buffer[epi_idx])
            while True:
                index = random.randint(0, current_epi_buffer_len)
                if index <= current_epi_buffer_len - self.traj_seq_len:
                    break
            list_.append(index)
        return list_
    
    def generate_epi_random_indices(self, batch_size):
        '''
            Output:
                list_: list, [batch_size,]: indices used for brits model training, determines what kind of trajectory.
        '''
        list_ = []
        for _ in range(batch_size):
            if self.epi_position == 0:
                epi_idx = 0
            elif self.epi_position > 0 and self.position <= self.traj_seq_len:
                epi_idx = random.randint(0, self.epi_position-1)
            else:
                epi_idx = random.randint(0, self.epi_position)
            list_.append(epi_idx)
        return list_

    def __len__(self):
        return len(self.buffer)
