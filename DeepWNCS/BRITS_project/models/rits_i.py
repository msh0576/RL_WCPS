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


class TemporalDecay(nn.Module):
    def __init__(self, input_size):
        super(TemporalDecay, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
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
        # input_size = 35
        if physics_info != None:
            _, spcf_obs_size, spcf_action_size = physics_info
            input_size = sum(spcf_obs_size) + sum(spcf_action_size)
        else:
            input_size = 12
        self.rnn_cell = nn.LSTMCell(input_size * 2, RNN_HID_SIZE)

        self.regression = nn.Linear(RNN_HID_SIZE, input_size)
        self.temp_decay = TemporalDecay(input_size = input_size)

        self.out = nn.Linear(RNN_HID_SIZE, 1)
    '''
    # Original
    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values'] # torch.Size([32 (batch_size), 49 (sequence len), 35 (LSTM output size or one vector size)])
        masks = data[direct]['masks']   # torch.Size([32, 49, 35])
        deltas = data[direct]['deltas'] # torch.Size([32, 49, 35])

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
        # print("labels:", labels)

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

            gamma = self.temp_decay(d)
            h = h * gamma
            x_h = self.regression(h)

            x_c =  m * x +  (1 - m) * x_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)   # torch.Size([32, 1]),  time sequnce 동안 model을 돌린 후, 제일 마지막 계산 후 h로 y를 추정함.
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)  # y_loss로는 모델을 업데이트 하는것 같지 않은데...

        # only use training labels
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + 0.1 *y_loss, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}
    '''
    # By sihoon
    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values'] # torch.Size([32 (batch_size), 49 (sequence len), 12 (LSTM output size or one vector size)])
        masks = data[direct]['masks']   # torch.Size([32, 49, 12])
        deltas = data[direct]['deltas'] # torch.Size([32, 49, 12])
        # print("values shape:", values.shape)    # torch.Size([1, 50, 12])

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        times = data[direct]['times']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
        # print("labels:", labels)

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

            gamma = self.temp_decay(d)
            h = h * gamma
            x_h = self.regression(h)

            x_c =  m * x +  (1 - m) * x_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)   # torch.Size([32, 1]),  time sequnce 동안 model을 돌린 후, 제일 마지막 계산 후 h로 y를 추정함.
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)  # y_loss로는 모델을 업데이트 하는것 같지 않은데...

        # only use training labels
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / self.seq_len, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks, 'times': times}
    
    def run_on_batch(self, data, optimizer):
        # print("data:", data['forward']['values'].shape)
        # print("data['forward']['values']:", data['forward']['values'])
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()  # x_loss의 MSE + 0.1*y_loss로 업데이트 됨.
            optimizer.step()

        return ret
    
