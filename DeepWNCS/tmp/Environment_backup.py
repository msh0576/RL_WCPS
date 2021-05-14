# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:46:27 2020

@author: Sihoon
"""

import matplotlib.pyplot as plt
import numpy as np
from Plant.pendulumSim import Pendulum
import Plant.pendulumParam as P

dtype = np.float32

THETA_ERROR_MAX = 10
THETA_ERROR_MIN = 0
AMPLITUDE = 0.25    # pendulum reference size
POSITION_ERROR_MAX = AMPLITUDE*2 + AMPLITUDE*0.2
POSITION_ERROR_MIN = 0
EPISODE_DURATION_MAX = P.t_end
EPISODE_DURATION_MIN = P.t_start
CONTROL_WEIGHT = 0
DURATION_WEIGHT = 0.5
UTILITY_WEIGHT = 0.5

class environment():
    def __init__(self, num_plant, pend_conf, purpose = 'train'):
        self.purpose = purpose
        self.num_plant = num_plant
        self.pend_conf = pend_conf
        # for test env.step()
        
    def reset(self):
        self.util_count = 0
        self.step_count = 0
        self.plants = []
        for i in range(self.num_plant):
            plant = Pendulum(network = 'wireless', pend_conf = self.pend_conf['pend_%s'%(i)])
            self.plants.append(plant)
        if self.purpose == 'test':  # append a wireline networked plant
            self.plants.append(Pendulum(network = 'wire', pend_conf = self.pend_conf['pend_%s'%(i)]))
        state = self.get_state()
        return state

    
    
    def step(self, action, t):
        '''
        - This function is invoked every 10 ms
        - Reflecting on a selected transmission schedule, the corresponding plant's status is updated
        
        '''
        self.step_count += 1
        schedule = float(action)/2
        # schedule = self.myNetworkScheduler.generate_networkAction() # for test
        
        for plant in self.plants:   # plants' state update
            # wireless network update
            if schedule == plant.mySensing: # sensing path
                plant.update_state_in_controller()
            elif schedule == plant.myActuating: # actuating path
                plant.update_command_in_plant()
            elif plant.network == 'wire': # wireline network update
                plant.update_state_in_controller()
                plant.update_command_in_plant()
            plant.time_step(t) # just state update
        
        next_state = self.get_state()       # get next env state (error vector)
        done = self.check_termination()     # check done
        reward = self.get_reward(t, done, schedule)          # get reward
        return next_state, reward, done, 0
    
    def get_state(self):
        '''
        [version 1]
        output: state (error vector) = (e^1, e^2, ... e^n)
        where e^i is a [2X1] vector b.t.w reference vector and current plant state
        reference vector = [[cart position], [theta]]
        ------------------------------------
        [version 2]
        output: (state vector) = (x^1, x^2, ..., x^n)
        where x^i is a [5X1] vector
        '''
        # compute plants' error
        # errorVector = self.get_errorVector()
        stateVector = self.get_stateVector()
        
        # generate state (error vector) from list to array
        state = np.array([], dtype=dtype)
        for i in range(self.num_plant):
            if state.size == 0:
                state = stateVector[i]
            else:
                state = np.concatenate((state, stateVector[i]), axis = 0) # Error vector
        
        return state
    
    def get_stateVector(self):
        '''
        output: x vector [4X1] + cart position reference [1X1] = [5X1]
        '''
        stateVector = []
        for plant in self.plants:
            if plant.network == 'wireless':
                x = plant.pendulum.state    # [4X1]
                ref = np.array([[plant.currReference()]])
                state = np.concatenate((x, ref), axis = 0)
                stateVector.append(state)
        return stateVector
                
    def get_errorVector(self):
        errorVector = []
        for plant in self.plants:
            if plant.network == 'wireless':
                # plant i's reference
                cartPosition_ref = plant.currReference()
                refVector = np.array([[cartPosition_ref], [0]]) # reference vector : position and theta : [2X1]
                # plant i's states
                x = plant.pendulum.state    # [4X1]
                plantError = np.array([[np.abs(refVector.item(0) - x.item(0))], [np.abs(refVector.item(1) - (180.0/np.pi * x.item(1)) )]])
                errorVector.append(plantError)
        
        return errorVector
    
    def get_reward(self, t, done, schedule):
        '''
        reward is sum of error of each plant
        '''
        errorVector = self.get_errorVector()
        Gamma = np.identity(2, dtype=dtype)
        reward = 0
        sum_error_max = POSITION_ERROR_MAX + THETA_ERROR_MAX
        sum_error_min = POSITION_ERROR_MIN + THETA_ERROR_MIN
        
        error_reward = 0
        duration_reward = 0
        util_reward = 0
        if not done:
            # reward relative to control error
            for error in errorVector:
                # sum_error = np.sum(error)
                # norm_sum_error = ((sum_error - sum_error_min) / (sum_error_max - sum_error_min)) / self.num_plant
                theta_error = error.item(1)
                norm_theta_error = ((theta_error - THETA_ERROR_MIN) / (THETA_ERROR_MAX - THETA_ERROR_MIN)) / self.num_plant
                error_reward -= abs(norm_theta_error)
            
            # reward relative to network utilization
            self.util_count += 1 if schedule == float(self.num_plant) else 0
            
            # reward relative to episode duration
            if round(t,2) == P.t_end - 0.01:    # episode final time
                # duration_reward = (t - EPISODE_DURATION_MIN) / (EPISODE_DURATION_MAX - EPISODE_DURATION_MIN)
                duration_reward += 1
                util_reward += self.util_count/self.step_count
        else:
            error_reward = -1
            duration_reward = -1
            # 중간에 끝날 시 만약 해당 슬롯을 스케줄 했더라면, 잘 될수도 있었기 때문에, 낭비되는 슬롯 이라는 의미에서 - 해준다.
            util_reward -= self.util_count/self.step_count  
            print("episode done!")
        
        reward += (CONTROL_WEIGHT * error_reward) + (DURATION_WEIGHT * duration_reward) + (UTILITY_WEIGHT * util_reward)
        return reward
    
    def update_plant_state(self, t):
        for plant in self.plants:
            plant.time_step(t)
    
    
    def check_termination(self):
        '''
        if the current theta and position of each plant gets out of THETA_ERROR_MAX and POSITION_ERROR_MAX,
        then the episode terminates
        '''
        done = False
        penalty = 0
        errorVector = self.get_errorVector()
        for error in errorVector:
            position_error = error[0][0]
            theta_error = error[1][0]
            if abs(position_error) > POSITION_ERROR_MAX:
                done = True
                break
            if abs(theta_error) > THETA_ERROR_MAX:
                done = True
                break
        return done
    
    def get_current_plant_status(self):
        r_buff = []
        x_buff = []
        u_buff = []
        for plant in self.plants:
            r_buff.append(plant.r)
            x_buff.append(plant.pendulum.state)
            u_buff.append(plant.prev_u)
        return r_buff, x_buff, u_buff
    
    def action_to_schedule(self, action):
        '''
        - action: neural network output index that has a maximum probability
        - action range: [0 - (2*num_plant - 1)]
        - Each plant has both sensing and actuating flows
        - if (action / 2) == x.0, then sensing flow is selected
        - if (action / 2) == x.5, then actuating flow is selected
        '''
        action_info = float(action)/2
        select_plant_id = int(action_info)
        select_flow = action_info - select_plant_id  
        
        return select_plant_id, select_flow
    
    def wire_step(self, t):
        for plant in self.plants:   # plants' state update
            # wireless network update
            plant.update_state_in_controller()
            plant.update_command_in_plant()
            plant.time_step(t) # just state update
        
