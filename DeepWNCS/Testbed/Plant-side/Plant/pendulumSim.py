import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')  # add parent directory
import Plant.pendulumParam as P
from Plant.signalGenerator import signalGenerator
from Plant.pendulumDynamics import pendulumDynamics


class Pendulum:
    def __init__(self, pend_conf):
        # instantiate pendulum, controller, and reference classes
        self.pendulum = pendulumDynamics()
        self.id = pend_conf['id']
        
        self.myActuating = float(self.id)
        
        self.y = self.pendulum.h()  # output of system at start of simulation
        self.prev_u = 0.0
        self.prev_x = self.pendulum.state
    
    def time_step(self, t):
        '''
        this function is invoked every 1ms
        change pendulum states
        '''
    
        # updates control and dynamics at faster simulation rate
        d = 0 # disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        self.y = self.pendulum.update(self.prev_u + d)  # propagate system


    def get_currState(self):
        self.prev_x = self.pendulum.state
        return self.prev_x


