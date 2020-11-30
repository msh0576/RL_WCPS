# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch

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


if __name__ == '__main__':
    set_cuda() 
    print("Hello!")