import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import BRITS_project.utils
import argparse
import BRITS_project.data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 80
# RNN_HID_SIZE = 64
RNN_HID_SIZE = 128

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size, spcf_obs_size=None, spcf_action_size=None):
        super(FeatureRegression, self).__init__()
        if spcf_obs_size != None and spcf_action_size != None:
            assert(len(spcf_obs_size)==len(spcf_action_size)), "FeatureRegression() Error in rits.py"
            self.build(input_size, spcf_obs_size, spcf_action_size)
        else:
            self.build(input_size)

    def build(self, input_size, spcf_obs_size=None, spcf_action_size=None):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        ## by sihoon
        if spcf_obs_size == None:
            m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        else:
            m = torch.zeros(input_size, input_size)
            num_plant = len(spcf_obs_size)
            for syst_idx in range(len(spcf_obs_size)):
                obs_start = sum(spcf_obs_size[:syst_idx])
                action_start = sum(spcf_obs_size[:]) + sum(spcf_action_size[:syst_idx])
                m[obs_start:obs_start+spcf_obs_size[syst_idx], obs_start:obs_start+spcf_obs_size[syst_idx]] = 1.
                m[obs_start:obs_start+spcf_obs_size[syst_idx], action_start: action_start+spcf_action_size[syst_idx]] = 1.
                m[action_start: action_start+spcf_action_size[syst_idx], action_start: action_start+spcf_action_size[syst_idx]] = 1.
            m -= torch.eye(input_size, input_size)

        # print("m:", m)
        self.register_buffer('m', m)

        self.reset_parameters()
        # print("self.W:", self.W, self.W.shape)
        # print("self.W * Variable(self.m):", self.W * Variable(self.m))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, physics_info=None):
        super(Model, self).__init__()
        if physics_info != None:
            self.build(physics_info)
        else:
            self.build()
        self.seq_len = seq_len

    def build(self, physics_info=None):
        spcf_obs_size, spcf_action_size = None, None
        if physics_info != None:
            num_plant, spcf_obs_size, spcf_action_size = physics_info
            input_size = sum(spcf_obs_size) + sum(spcf_action_size)
        else:
            input_size = 12
            # input_size = 35
        input_size = input_size

        self.rnn_cell = nn.LSTMCell(input_size * 2, RNN_HID_SIZE)
        # self.rnn_cell = nn.RNNCell(input_size * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = input_size, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = input_size, output_size = input_size, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, input_size)
        self.feat_reg = FeatureRegression(input_size, spcf_obs_size, spcf_action_size)

        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(RNN_HID_SIZE, 1)

    '''
    # Original
    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + y_loss * 0.3, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}
    '''

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        times = data[direct]['times']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            # c_h = x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))
            # h = self.rnn_cell(inputs, h)

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / self.seq_len, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks, 'times': times}


    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
