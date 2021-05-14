import numpy as np
from Util.utils import to_tensor, set_cuda
from Environments.Wire_Environment import wire_environment
from MultipendulumSim_PETS import MultipendulumSim_PETS
import torch

path = '/home/sihoon/works/RL_WCPS-master/DeepWNCS/Inverted_Pendulum_Simulation'
Log_paths = {
            'dataset_path': path + '/LogData/log/PET_param_.pickle',
            'model_path': path + '/LogData/model/PET_model_.pickle'
            }
Log_paths2 = {  # estimate_horizon_costs_with_model 에서 cost 식 변경
            'dataset_path': path + '/LogData/log/PET_param_2.pickle',
            'model_path': path + '/LogData/model/PET_model_2.pickle'
            }

def readArgs(num_plant):
    conf = {
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': 1,
        "num_train_iter": 10,
        "num_epi_per_iter": 1,
        "num_model_warmup": 50,
        "num_eval": 1,
        # "memory_capacity": 10000,
        # "batch_size": 100,
        # "gamma": 0.99,  # discount factor
        # "learning_rate": 1e-4,
        # "epsilon_start": 1,
        # "epsilon_end": 0.02,
        # "epsilon_decay": 1000,
        # "target_update": 10,
        # "update_iteration": 200,
        "horizon_len": 10,
        "network_size": 5,
        "elite_size": 5,
        ### CEM params ###
        "pop_size": 50,
        "max_iters": 5,
        "num_particles": 4
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

    PETS = MultipendulumSim_PETS(args, cuda_info, env, Log_paths2)
    PETS.train(cont=False)
    # PETS.evaluate()
    # PETS.check_training_result()


if __name__ == '__main__':
    main()
    



