# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:52:26 2020

@author: Sihoon
"""
from Environment import environment
from MultipendulumSim_A2C import MultipendulumSim_A2C
from MultipendulumSim_DQN import MultipendulumSim_DQN
from MultipendulumSim_RDPG import MultipendulumSim_RDPG

A2C_LOG = './LogData/log/A2C_log.pickle'
A2C_LOG_v2 = './LogData/log/A2C_v2_log.pickle'
A2C_LOG_v3 = './LogData/log/A2C_v3_log.pickle'
A2C_LOG_v4 = './LogData/log/A2C_v4_log.pickle'
A2C_LOG_v5 = './LogData/log/A2C_v5_log.pickle'

ACTOR_MODEL = './LogData/model/actor.pickle'
ACTOR_MODEL_v2 = './LogData/model/actor_cont.pickle'
ACTOR_MODEL_v3 = './LogData/model/actor_dura_cont.pickle'
ACTOR_MODEL_v4 = './LogData/model/actor_longdura_cont.pickle'
ACTOR_MODEL_v5 = './LogData/model/actor_longdura_cont_statechange.pickle'
CRITIC_MODEL = './LogData/model/critic.pickle'


if __name__ == '__main__': 
    # =============================================================================
    # configuration
    # =============================================================================
    num_plant = 3
    configuration = {
        'num_plant' : num_plant,
        'state_dim' : 2*num_plant,
        'action_dim': num_plant+1,
        "num_episode": 500,
        "memory_capacity": 10000,
        "batch_size": 32,
        "gamma": 0.99,  # discount factor
        "learning_rate": 1e-4,
        "epsilon_start": 1,
        "epsilon_end": 0.02,
        "epsilon_decay": 1000,
        "target_update": 10
        }
    pend_configuration = {}
    # amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.15, 0.2, 0.2, 0.2]
    # trigger_list = [100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100, 200, 300, 400, 500]  # ms
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [100, 200, 300, 400, 500]  # ms
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
    # multiPendulumSim_A2C.train(ACTOR_MODEL_v2, A2C_LOG_v2)
    # multiPendulumSim_A2C.train_v2(ACTOR_MODEL_v2)
    # multiPendulumSim_A2C.plot_cumulate_reward(A2C_LOG_v3)
    
    
    # test
    new_env = environment(num_plant, pend_configuration)
    multiPendulumSim_A2C.test(new_env, iteration=100, test_duration = 20, model_path = ACTOR_MODEL_v2, algorithm = 'A2C')
    
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



