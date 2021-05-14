
import torch, pickle
from torch.autograd import Variable
import numpy as np

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