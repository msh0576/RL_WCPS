# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:44:07 2020

@author: Sihoon
"""
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.lines import Line2D

class testDataPlotter:
    def __init__(self, conf):
        self.conf = conf
        self.num_rows = 3    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns
        
        # pendulum history
        self.pendulums = {}
        for idx in range(conf['num_plant']):
            self.pendulums['pend_%s'%(idx)] = {'time_history': [],  # time
                                          'zref_history': [],       # reference position z_r
                                          'z_history': [],          # position z
                                          'theta_history': [],      # angle theta
                                          'Force_history': []}      # control force
        self.error_history = {}
        for idx in range(conf['num_plant']):
            self.error_history['for_pend_%s'%(idx)] = []
            
        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)
        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='z(m)', title='Pendulum Data') )
        self.handle.append(myPlot(self.ax[1], ylabel='theta(deg)'))
        self.handle.append(myPlot(self.ax[2], xlabel='t(s)', ylabel='error'))

    
    def update(self, pend_idx, t, reference, states, ctrl):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        self.pendulums['pend_%s'%(pend_idx)]['time_history'].append(t)
        self.pendulums['pend_%s'%(pend_idx)]['zref_history'].append(reference)
        self.pendulums['pend_%s'%(pend_idx)]['z_history'].append(states.item(0))
        self.pendulums['pend_%s'%(pend_idx)]['theta_history'].append(180.0/np.pi*states.item(1))
        self.pendulums['pend_%s'%(pend_idx)]['Force_history'].append(ctrl)
        
    def plot(self):
        # update the plots with associated histories
        self.handle[0].update(self.pendulums['pend_%s'%(0)]['time_history'], 
                              [self.pendulums['pend_%s'%(0)]['z_history'],
                                self.pendulums['pend_%s'%(1)]['z_history'],
                                self.pendulums['pend_%s'%(2)]['z_history'],
                                # self.pendulums['pend_%s'%(self.conf['num_plant'])]['z_history'],
                                self.pendulums['pend_%s'%(0)]['zref_history']])
        self.handle[1].update(self.pendulums['pend_%s'%(0)]['time_history'], 
                              [self.pendulums['pend_%s'%(0)]['theta_history'],
                                self.pendulums['pend_%s'%(1)]['theta_history'],
                                self.pendulums['pend_%s'%(2)]['theta_history']
                                # self.pendulums['pend_%s'%(self.conf['num_plant'])]['theta_history']
                                ])
        
        # optimlaError between wire and wireless systems
        # error0 = self.pendulums['pend_%s'%(self.conf['num_plant'])]['z_history'][-1] - self.pendulums['pend_%s'%(0)]['z_history'][-1]
        # error1 = self.pendulums['pend_%s'%(self.conf['num_plant'])]['z_history'][-1] - self.pendulums['pend_%s'%(1)]['z_history'][-1]
        # error2 = self.pendulums['pend_%s'%(self.conf['num_plant'])]['z_history'][-1] - self.pendulums['pend_%s'%(2)]['z_history'][-1]
        # self.error_history['for_pend_0'].append(np.abs(error0))
        # self.error_history['for_pend_1'].append(np.abs(error1))
        # self.error_history['for_pend_2'].append(np.abs(error2))
        # self.handle[2].update(self.pendulums['pend_%s'%(0)]['time_history'], 
        #                       [self.error_history['for_pend_0'],
        #                        self.error_history['for_pend_1'],
        #                        self.error_history['for_pend_2']])
    def print_totalError(self):
        totalError = 0
        for idx in range(self.conf['num_plant']):
            totalError += sum(self.error_history['for_pend_%s'%(idx)])
        print("Total Error relative to optimal control:", totalError)
    
    def print_schedRatio(self, episode_actions, episode_step_count):
        '''
        - input 'episode_actions' is an action history list for all episodes, 'episode_step_count' is a list for all episodes
        - according to action values, a plant has 1 if its actuating flow is scheduled, -1 if its sensing flow is scheduled,
        0 if no flows are scheduled at that time slot.
        '''
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        total_schedRatio = {}
        for idx in range(self.conf['num_plant']):
                total_schedRatio['pend_%s'%(idx)] = {'sensRatio': [],
                                                     'actuRatio': []}
        total_schedRatio['utility'] = []
                
        for epi in range(len(episode_actions)):
            plants_schedule = {}
            count_empty_sched = 0
            for idx in range(self.conf['num_plant']):
                plants_schedule['pend_%s'%(idx)] = []
            for action in episode_actions[epi]:
                schedule = float(action)
                pend_id = int(schedule)
                flow_id = schedule - pend_id
                for idx in range(self.conf['num_plant']):
                    if pend_id == idx and flow_id == 0.:   # actuating path
                        plants_schedule['pend_%s'%(idx)].append(1)
                    elif pend_id == idx and flow_id == 0.5: # sensing path
                        plants_schedule['pend_%s'%(idx)].append(-1)
                    else:
                        plants_schedule['pend_%s'%(idx)].append(0)
                count_empty_sched += 1 if schedule == self.conf['num_plant'] else 0     # count empty schedule
            # compute average flow ration for an episode                
            for idx in range(self.conf['num_plant']):
                sensRatio = plants_schedule['pend_%s'%(idx)].count(-1)/episode_step_count[epi]*100
                actuRatio = plants_schedule['pend_%s'%(idx)].count(1)/episode_step_count[epi]*100
                # print('pend_%s sensing flow ration:%s'%(idx, sensRatio))
                # print('pend_%s actuating flow ration:%s'%(idx, actuRatio))
                total_schedRatio['pend_%s'%(idx)]['sensRatio'].append(sensRatio)
                total_schedRatio['pend_%s'%(idx)]['actuRatio'].append(actuRatio)
            # print('network utility: ', count_empty_sched/episode_step_count[epi]*100)
            total_schedRatio['utility'].append(count_empty_sched/episode_step_count[epi]*100)
        # print average results
        for idx in range(self.conf['num_plant']):
            print('pend_%s average sensing flow ration:%s'%(idx, sum(total_schedRatio['pend_%s'%(idx)]['sensRatio'])/len(total_schedRatio['pend_%s'%(idx)]['sensRatio'])))
            print('pend_%s average actuating flow ration:%s'%(idx, sum(total_schedRatio['pend_%s'%(idx)]['actuRatio'])/len(total_schedRatio['pend_%s'%(idx)]['actuRatio'])))
        print('average network utility: ', sum(total_schedRatio['utility'])/len(total_schedRatio['utility']))
        # plt.figure(num=2)
        # for idx in range(self.conf['num_plant']):
        #     plt.plot(plants_schedule['pend_%s'%(idx)], label='pend_%s'%(idx), color = colors[idx])
        # plt.legend()
        # plt.title('Schedule')
        # plt.show()
        
        # print('Press key to close')
        # plt.waitforbuttonpress()
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