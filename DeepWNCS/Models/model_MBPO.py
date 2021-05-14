import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gzip
from Util.utils import to_tensor, to_numpy

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Game_model(nn.Module):
    def __init__(self, state_size, action_size, reward_size, device, hidden_size=200, learning_rate=1e-2):
        super(Game_model, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            Swish()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )
        self.nn4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = nn.Linear(hidden_size, self.output_dim * 2)

        self.max_logvar = Variable(torch.ones((1, self.output_dim)).type(torch.FloatTensor) / 2, requires_grad=True).to(device)
        self.min_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 10, requires_grad=True).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        '''
            output  : [batch_size, (state_size + reward_size)*2 = (mean:var)]
            mean    : [batch_size, state_size + reward_size]
            var     : [batch_size, state_size + reward_size]
        '''
        nn1_output = self.nn1(x)
        nn2_output = self.nn2(nn1_output)
        nn3_output = self.nn3(nn2_output)
        nn4_output = self.nn4(nn3_output)
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, torch.exp(logvar)

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        '''
            mse_loss, var_loss  : scalar value
        '''
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            mse_loss = torch.mean(torch.pow(mean - labels, 2) * inv_var)
            var_loss = torch.mean(logvar)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = nn.MSELoss()
            total_loss = mse_loss(input=logits, target=labels)
        return total_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()

class Ensemble_Model():
    def __init__(self, network_size, elite_size, state_size, action_size, device, reward_size=1, hidden_size=200):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        for i in range(network_size):
            self.model_list.append(Game_model(state_size, action_size, reward_size, device, hidden_size))

    def train(self, inputs, labels, batch_size=256):
        '''
            labels  : [memory_len, (reward_size + state_size)]  where state is a delta_state (next_state - state)

        '''
        for start_pos in range(0, inputs.shape[0], batch_size):
            input = to_tensor(inputs[start_pos : start_pos + batch_size])
            label = to_tensor(labels[start_pos : start_pos + batch_size])
            losses = []
            for model in self.model_list:
                # train model via maximum likelihood - refer to the paper
                mean, log_var = model(input)
                loss = model.loss(mean, log_var, label)
                model.train(loss)
                losses.append(loss)
        sorted_loss_idx = np.argsort(losses)
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

    def predict(self, inputs, batch_size=1024):
        '''
            it arranges the outputs (mean, var) from each of the ensemble models
            in each row of the ensemble_mean/var matrix
            ------------------------

            output: ensemble_mean and ensemble_logvar size = [network_size, batch_size, state_dim + reward_dim]
                where network_size means # of model in a model_list
                        output state means delta_state, not next_state
        '''
        ensemble_mean = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        ensemble_logvar = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        for i in range(0, inputs.shape[0], batch_size):
            input = to_tensor(inputs[i:min(i + batch_size, inputs.shape[0])])
            for idx in range(self.network_size):
                pred_2d_mean, pred_2d_logvar = self.model_list[idx](input)
                ensemble_mean[idx,i:min(i + batch_size, inputs.shape[0]),:], ensemble_logvar[idx,i:min(i + batch_size, inputs.shape[0]),:] \
                    = to_numpy(pred_2d_mean.detach()), to_numpy(pred_2d_logvar.detach())

        return ensemble_mean, ensemble_logvar


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
