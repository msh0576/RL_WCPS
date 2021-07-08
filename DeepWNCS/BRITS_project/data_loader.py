import os
import time

import ujson as json
#import json
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, path=None, content=None):
        super(MySet, self).__init__()
        if path == None:
            path = './BRITS_project/json/sac_dataset_randomTraj.pickle'
        # self.content = open('./json/json').readlines()
        if content == None:
            with open(path, 'rb') as pickle_file:  # by sihoon
                self.content = pickle.load(pickle_file)
        else:
            self.content = content

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        # rec = json.loads(self.content[idx])
        rec = self.content[idx]   # by sihoon
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def to_tensor_dict(recs):
    values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
    masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
    deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
    forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

    evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
    eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

    times = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['times'], r)), recs)))    # by sihoon

    return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks, 'times': times}
    # return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))
    
    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def get_loader(batch_size = 64, shuffle = True, path=None):
    data_set = MySet(path)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
