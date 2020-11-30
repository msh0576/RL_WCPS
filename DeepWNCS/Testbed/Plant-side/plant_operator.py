# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:42:04 2020

@author: Sihoon
"""
from datetime import datetime
import time, threading
import numpy as np
import matplotlib.pyplot as plt
from Plant.pendulumSim import Pendulum
from Plotter import plotter, myPlot
import process_timer
import multiprocessing
import ctypes
import random



class pendulum_manager(multiprocessing.Process):
    def __init__(self, mprun_break, conf, pend_conf, shared_dict, shared_lists):
        multiprocessing.Process.__init__(self)
        self.shared_dict = shared_dict
        self.shared_lists = shared_lists
        self.mprun_break = mprun_break
        self.num_plant = conf['num_plant']
        
        # generate pendulums
        self.pendulums = []
        for idx in range(self.num_plant):
            pendulum = Pendulum(pend_conf['pend_%s'%(idx)])
            self.pendulums.append(pendulum)
        
    
    def run(self):
        print("run here")
        self.start_time = round(time.time(), 3)
        print("start time:", self.start_time)
        self.timer()
        
    def timer(self):
        '''
        every 1ms, pendulum's state is changed
        '''
        try:
            while not self.mprun_break.value:
                second = round(time.time(), 3)
                millisecond = int(round(time.time() * 1000))
                plot_period = 100 # ms
                print("second:", second)
                # step pendulum's state
                state_buff = []
                for system in self.pendulums:
                    system.time_step(second)
                    plant_state = system.get_currState()
                    self.update_dataPlot(self.shared_lists[system.id], system.id, round(second - self.start_time, 3), plant_state)
                time.sleep(1)
        finally:
            print("process 1 finish")
        
            
    def update_dataPlot(self, shared_list, pend_idx, t, states):
        '''
            Add to the time and data histories, and update the plots.
        buffer list : [time, position, theta]
        '''
        # update the time history of all plot variables
        buff = []
        buff.append(t)
        buff.append(states.item(0) + random.uniform(0,1))
        buff.append(180.0/np.pi*states.item(1))
        shared_list.append(buff)


def init_plot_dict(num_plant, manager):
    shared_dict = manager.dict()
    shared_lists = [manager.list() for i in range(num_plant)]
    
    for idx in range(num_plant):
        shared_dict['%s'%(idx)] = shared_lists[idx]
    
    return shared_dict, shared_lists



    
if __name__ == '__main__':
    num_plant = 5
    testing_interval = 5 # second
    configuration = {
        'num_plant' : num_plant,
        'plant_state_dim' : 4,
        'state_dim' : 2*num_plant+1,
        'action_dim': num_plant+1
        }
    pend_configuration = {}
    for i in range(num_plant):
        pend_configuration['pend_%s'%(i)] = {'id': i}
        

    # pendulum_manager = pendulum_manager(configuration, pend_configuration)
    
    
    mps = list()
    manager = multiprocessing.Manager()
    shared_dict, shared_lists = init_plot_dict(num_plant, manager)
    mprun_break = manager.Value(ctypes.c_bool, False)
    
    mp1 = pendulum_manager(mprun_break, configuration, pend_configuration, shared_dict, shared_lists)
    mp1.daemon = True
    mp2 = plotter(mprun_break, configuration, shared_dict, shared_lists)
    mp2.daemon = True
    mps.append(mp1)
    mps.append(mp2)
    
    print(" befor start ")
    for mp in mps:
        mp.start()
    
    time.sleep(testing_interval)
    mprun_break.value = True
    
    print("before join")
    for mp in mps:
        mp.join()
    
    print("done")
    
    
    
    