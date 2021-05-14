import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')  # add parent directory
import Plant.pendulumParam as P
from Plant.signalGenerator import signalGenerator
from Plant.pendulumDynamics import pendulumDynamics
from Plant.pendulumController import pendulumController


class Pendulum:
    def __init__(self, network, pend_conf):
        # instantiate pendulum, controller, and reference classes
        self.id = pend_conf['id']
        self.pend_conf = pend_conf
        disturbance = signalGenerator(amplitude=0.1)
        self.network = network
        self.trigger_time = pend_conf['trigger_time']

        # for scheduling wireless networks
        self.mySensing = float(self.id) + 0.5
        self.myActuating = float(self.id)

        self.reset()


    def reset(self):
        self.pendulum = pendulumDynamics()
        self.controller = pendulumController()
        self.reference = signalGenerator(amplitude=self.pend_conf['amplitude'], frequency=self.pend_conf['frequency'])

        self.y = self.pendulum.h()  # output of system at start of simulation
        self.prev_u = 0.0
        self.prev_x = self.pendulum.state
        self.r = self.reference.square(t=0) # initial reference

        return self.prev_x

    def time_step(self, t):
        '''
        this function is invoked every 1ms
        change pendulum states
        '''

        # updates control and dynamics at faster simulation rate
        self.r = self.reference.square(t)  # reference input
        d = 0 # disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        self.y = self.pendulum.update(self.prev_u + d)  # propagate system

    def update_state_in_controller(self):
        x = self.pendulum.state
        self.prev_x = x

    def update_command_in_plant(self):
        u = self.controller.update(self.r, self.prev_x)
        self.prev_u = u

    def currReference(self):
        return self.r

    def get_currState(self):
        return self.prev_x

    def update_command_in_plant_DDPG(self, action):
        self.prev_u = action

    # def time_step(self, t):
    #     '''
    #     this function is invoked every 1ms
    #     input: action,
    #     change pendulum states
    #     '''
    #     # Propagate dynamics in between plot samples
    #     t_next_plot = t + P.t_plot

    #     # updates control and dynamics at faster simulation rate

    #     while t < t_next_plot:
    #         self.r = self.reference.square(t)  # reference input
    #         d = 0 # disturbance.step(t)  # input disturbance
    #         n = 0.0  #noise.random(t)  # simulate sensor noise


    #         if self.network == 'wire':  # wireline networked control
    #             x = self.pendulum.state
    #             self.prev_x = x
    #             u = self.controller.update(self.r, x)  # update controller
    #             self.prev_u = u
    #         else:   # wireless networked control
    #             if round(t,3)*1000 % 10 == 0:   # every 10 ms
    #                 schedule = self.myNetworkScheduler.generate_networkAction()
    #                 if schedule == self.mySensing: # sensing path delivery
    #                     x = self.pendulum.state
    #                     self.prev_x = x
    #                 elif schedule == self.myActuating: # actuating path delivery
    #                     u = self.controller.update(self.r, self.prev_x)
    #                     self.prev_u = u

    #         self.y = self.pendulum.update(self.prev_u + d)  # propagate system
    #         t = t + P.Ts  # advance time by Ts
    #     # update animation and data plots
    #     # self.animation.update(self.pendulum.state)
    #     # self.dataPlot.update(t, r, self.pendulum.state, u)
    #     # plt.pause(0.0001)
    #     return t, self.r, self.pendulum.state, self.prev_u
