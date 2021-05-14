# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:52:26 2020

@author: Sihoon
"""
from Environments.Environment import environment
from MultipendulumSim_A2C import MultipendulumSim_A2C
from MultipendulumSim_DQN import MultipendulumSim_DQN
from MultipendulumSim_RDPG_v2 import MultipendulumSim_RDPG_v2
path = '/home/sihoon/works/RL_WCPS-master/DeepWNCS/Inverted_Pendulum_Simulation'

A2C_LOG_duration_control = path + '/LogData/log/A2C_log.pickle'
A2C_LOG_only_duration = path + '/LogData/log/A2C_log_only_duration.pickle'

A2C_MODEL_duration_control = path + '/LogData/model/actor.pickle'
A2C_MODEL_only_duration = path + '/LogData/model/actor_only_duration.pickle'


if __name__ == '__main__':
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 5
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 5*num_plant,
        'action_dim': (2*num_plant)+1,
        "num_episode": 1500,
        "memory_capacity": 10000,
        "batch_size": 32,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.005,
        "epsilon_decay": 1000,
        "target_update": 10
        }
    pend_configuration = {}
    # amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # trigger_list = [100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100, 200, 300, 400, 500]  # ms
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [10, 10, 10, 10, 10]  # ms
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i,
                                             'amplitude': amplitude_list[i],
                                             'frequency': frequency_list[i],
                                             'trigger_time': trigger_list[i]}


    # create environments
    env = environment(num_plant, pend_configuration)

    # =============================================================================
    # A2C
    # =============================================================================

    # train
    multiPendulumSim_A2C = MultipendulumSim_A2C(env, configuration)
    multiPendulumSim_A2C.train(A2C_MODEL_only_duration, A2C_LOG_only_duration)
    # multiPendulumSim_A2C.train_v2(ACTOR_MODEL_v2)
    # multiPendulumSim_A2C.plot_cumulate_reward(A2C_LOG_only_duration)


    # test
    # new_env = environment(num_plant, pend_configuration)
    # multiPendulumSim_A2C.test(new_env, iteration=100, test_duration = 20, model_path = ACTOR_MODEL_v2, algorithm = 'A2C')

    # =============================================================================
    # DQN
    # =============================================================================

    # train
    # multiPendulumSim_DQN = MultipendulumSim_DQN(env, configuration)
    # multiPendulumSim_DQN.train()
    # multiPendulumSim_DQN.plot_cumulate_reward()

    # test
    # new_env = environment(num_plant, pend_configuration, purpose = 'test')
    # multiPendulumSim_DQN.test(new_env, iteration=20, test_duration = 30, algorithm = 'A2C')


    # =============================================================================
    # RDPG
    # =============================================================================

    # train
    # multiPendulumSim_RDPG = MultipendulumSim_RDPG(env, configuration)
    # multiPendulumSim_RDPG.train()
