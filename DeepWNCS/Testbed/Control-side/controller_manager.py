# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:19:05 2020

@author: Sihoon
"""
import numpy as np
from Plant.pendulumController import pendulumController
from Plant.signalGenerator import signalGenerator
import Plant.pendulumParam as P

dtype=np.float32

class plant_controller:
    '''
    for a physical system
    '''
    def __init__(self, pend_conf):
        self.id = pend_conf['id']
        self.reference = signalGenerator(amplitude=pend_conf['amplitude'], frequency=pend_conf['frequency'])
        self.controller = pendulumController()
        
        self.r = self.reference.square(t=0) # initial reference
        # initial state
        self.prev_x = np.array([
            [P.z0],  # z initial position
            [P.theta0],  # Theta initial orientation
            [P.zdot0],  # zdot initial velocity
            [P.thetadot0],  # Thetadot initial velocity
        ])
    
    def time_step(self, state):
        '''
        every time_step, reference is changed
        input state : [5X1] it includes a reference value
        '''
        # self.prev_x = state
        pass
    
    def ref_step(self, t):
        self.r = self.reference.square(t)  # reference input
    
    def generate_command(self, plant_state):
        '''
        plant_state: shape [4X1], 
        '''
        self.prev_x = plant_state
        u = self.controller.update(self.r, self.prev_x)
        return u

class controller_manager:
    def __init__(self, conf, pends_conf):
        self.plant_state_dim = conf['plant_state_dim']
        self.num_plant = conf['num_plant']
        # generate controllers
        self.manager = []
        for idx in range(self.num_plant):
            system = plant_controller(pends_conf['pend_%s'%(idx)])
            self.manager.append(system)
        
    
    def get_commands(self, plants_state):
        '''
        assume input state shape: (plant_state_dim * num_plant,)
        output: commandVector 
        '''
        commandVector = []
        # get control command for each plant
        for idx in range(self.num_plant):
            system = self.manager[idx]
            plant_state = plants_state[self.plant_state_dim*idx : (self.plant_state_dim*idx)+self.plant_state_dim] # (4,)
            plant_state = plant_state.reshape(self.plant_state_dim, 1) # (4,1)
            command = system.generate_command(plant_state)  # float
            commandVector.append(command)
        return commandVector
    
    def ref_update(self, t):
        '''
        update reference according to current time for all plants
        '''
        for system in self.manager:
            system.ref_step(t)
    
    def get_errorVector(self, plants_state):
        '''
        plants_state : shape (4 * num_plant,)
        output: 
        '''
        errorVector = []
        
        for idx in range(self.num_plant):
            # plant i's reference
            cartPosition_ref = self.manager[idx].r
            reference = np.array([[cartPosition_ref], [0]], dtype=dtype)  # [2X1]
            # plant i's state
            x = plants_state[(self.plant_state_dim*idx):(self.plant_state_dim*idx)+self.plant_state_dim]
            x = x.reshape(self.plant_state_dim,1) # [4X1]
            plantError = np.array([[np.abs(reference.item(0) - x.item(0))], [np.abs(reference.item(1) - (180.0/np.pi * x.item(1)) )]], dtype=dtype)
            errorVector.append(plantError)
        return errorVector
    
    
    def get_state(self, plants_state):
        '''
        input: plants_state - shape (4 * num_plant,)
        output: state (error vector + time scalar) = (e^1, e^2, ... e^n, t)
        where e^i is a [2X1] vector b.t.w reference vector and current plant state
        reference vector = [[cart position], [theta]]
        '''
        # compute plants' error
        errorVector = self.get_errorVector(plants_state)
        # stateVector = self.get_stateVector()
        # generate state (error vector) from list to array
        state = np.array([], dtype=np.float32)
        for i in range(self.num_plant):
            if state.size == 0:
                # state = stateVector[i]
                state = errorVector[i]
            else:
                # state = np.concatenate((state, stateVector[i]), axis = 0) # Error vector
                state = np.concatenate((state, errorVector[i]), axis = 0) # Error vector
        state = state.squeeze(1)
        return state
    
    
    