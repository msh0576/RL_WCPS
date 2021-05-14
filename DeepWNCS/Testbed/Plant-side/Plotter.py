# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:43:14 2020

@author: Sihoon
"""

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.lines import Line2D
import multiprocessing
import time

class plotter(multiprocessing.Process):
    def __init__(self, mprun_break, conf, shared_dict, shared_lists):
        multiprocessing.Process.__init__(self)
        self.mprun_break = mprun_break
        self.shared_dict = shared_dict
        self.shared_lists = shared_lists
        self.conf = conf
        self.num_rows = 2    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns
        
    
    def init_plot(self):
        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)
        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='z(m)', title='Pendulum Data') )
        self.handle.append(myPlot(self.ax[1], xlabel='t(s)', ylabel='theta(deg)'))
        
        self.pendulums = {}
        for idx in range(self.conf['num_plant']):
            self.pendulums['pend_%s'%(idx)] = {'time_history': [],  # time
                                                      'zref_history': [],       # reference position z_r
                                                      'z_history': [],          # position z
                                                      'theta_history': [],      # angle theta
                                                      'Force_history': []}      # control force
        
    def run(self):
        '''
        every few seconds, plots the figure
        '''
        self.init_plot()
            
        try:
            while not self.mprun_break.value:
                self.process_list(self.shared_lists)
                self.plot()
                time.sleep(1)
        finally:
            print('Press key to close')
            plt.waitforbuttonpress()
            print("process 2 finish")
    
    def process_list(self, input_lists):
        '''
        lists: [[plant-1 list], [plant-2 list], ..., [plant-n list]]
        input a list shape: [[time, x(0)-position, x(1)-theta], [next time, .., ..]]
        output a list shape: [[time, next time, ...], [x(0), next x(0), ...], [x(1), next x(1), ...]]
        '''
        # figure axis
        for idx in range(self.conf['num_plant']):
            self.pendulums['pend_%s'%(idx)]['time_history'] = []
            self.pendulums['pend_%s'%(idx)]['z_history'] = []
            self.pendulums['pend_%s'%(idx)]['theta_history'] = []
            for list_ in input_lists[idx]:
                self.pendulums['pend_%s'%(idx)]['time_history'].append(list_[0])
                self.pendulums['pend_%s'%(idx)]['z_history'].append(list_[1])
                self.pendulums['pend_%s'%(idx)]['theta_history'].append(list_[2])
        
        
        
        

    def plot(self):
        # update the plots with associated histories
        self.handle[0].update(self.pendulums['pend_%s'%(0)]['time_history'], 
                              [self.pendulums['pend_%s'%(0)]['z_history']
                                # self.pendulums['pend_%s'%(1)]['z_history']
                                # self.pendulums['pend_%s'%(2)]['z_history'],
                                # self.pendulums['pend_%s'%(0)]['zref_history']
                                ])
        self.handle[1].update(self.pendulums['pend_%s'%(0)]['time_history'], 
                              [self.pendulums['pend_%s'%(0)]['theta_history']
                                # self.pendulums['pend_%s'%(1)]['theta_history']
                                # self.pendulums['pend_%s'%(2)]['theta_history']
                                ])
        plt.pause(0.0001)
        
class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, episode, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(episode,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(episode)
                self.line[i].set_ydata(data[i])
        
        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()