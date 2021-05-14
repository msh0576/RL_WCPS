import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter_v2:
    '''
        This class plots the time histories for the pendulum data.
    '''

    def __init__(self, conf, num_row=4, num_col=1):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = num_row    # Number of subplot rows
        self.num_cols = num_col    # Number of subplot columns
        self.conf = conf

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # pendulum history
        self.pendulums = {}
        for idx in range(conf['num_plant']):
            self.pendulums['pend_%s'%(idx)] = {'time_history': [],  # time
                                          'zref_history': [],       # reference position z_r
                                          'z_history': [],          # position z
                                          'theta_history': [],      # angle theta
                                          'Force_history': [],      # control force
                                          'reward': []
                                          }      
        self.error_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='z(m)', title='Pendulum Data'))
        self.handle.append(myPlot(self.ax[1], ylabel='theta(deg)'))
        self.handle.append(myPlot(self.ax[2], xlabel='t(s)', ylabel='force(N)'))
        self.handle.append(myPlot(self.ax[3], xlabel='t(s)', ylabel='reward'))
    

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
                               # self.pendulums['pend_%s'%(1)]['z_history'],
                               # self.pendulums['pend_%s'%(2)]['z_history'],
                                # self.pendulums['pend_%s'%(2)]['z_history'],
                                self.pendulums['pend_%s'%(0)]['zref_history']]
                              )
        self.handle[1].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['theta_history'],
                               # self.pendulums['pend_%s'%(1)]['theta_history'],
                               # self.pendulums['pend_%s'%(2)]['theta_history'],
                                # self.pendulums['pend_%s'%(2)]['theta_history']
                                ]
                              )
        self.handle[2].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['Force_history'],
                               # self.pendulums['pend_%s'%(1)]['Force_history'],
                               # self.pendulums['pend_%s'%(2)]['Force_history']
                                # self.pendulums['pend_%s'%(2)]['Force_history']
                                ]
                              )

        # calculate error between wire and wireless control
        # error = self.pendulums['pend_%s'%(2)]['z_history'][-1] - self.pendulums['pend_%s'%(1)]['z_history'][-1]
        # self.error_history.append(np.abs(error))
        # self.handle[3].update(self.pendulums['pend_%s'%(1)]['time_history'],
        #                       [self.error_history]
        #                       )
        plt.pause(0.0001)

    def close(self):
        plt.close()
    
    def plt_waitforbuttonpress(self):
        print('Press key to close')
        plt.waitforbuttonpress()
    



class dataPlotter_PETS(dataPlotter_v2):
    def __init__(self, conf):
        super().__init__(conf)
    
    def figure_sum_reward(self, x_data, y_data):
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)
        ax.set_xlabel('episode step')
        ax.set_ylabel('accumulated reward')


class dataPlotter_MPC(dataPlotter_v2):
    def __init__(self, conf):
        super().__init__(conf)
    
    def update(self, pend_idx, t, reference, states, ctrl, reward):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        self.pendulums['pend_%s'%(pend_idx)]['time_history'].append(t)
        self.pendulums['pend_%s'%(pend_idx)]['zref_history'].append(reference)
        self.pendulums['pend_%s'%(pend_idx)]['z_history'].append(states.item(0))
        self.pendulums['pend_%s'%(pend_idx)]['theta_history'].append(180.0/np.pi*states.item(1))
        self.pendulums['pend_%s'%(pend_idx)]['Force_history'].append(ctrl)
        self.pendulums['pend_%s'%(pend_idx)]['reward'].append(reward)

    def plot(self):
        # update the plots with associated histories
        self.handle[0].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['z_history'],
                               # self.pendulums['pend_%s'%(1)]['z_history'],
                               # self.pendulums['pend_%s'%(2)]['z_history'],
                                # self.pendulums['pend_%s'%(2)]['z_history'],
                                self.pendulums['pend_%s'%(0)]['zref_history']]
                              )
        self.handle[1].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['theta_history'],
                               # self.pendulums['pend_%s'%(1)]['theta_history'],
                               # self.pendulums['pend_%s'%(2)]['theta_history'],
                                # self.pendulums['pend_%s'%(2)]['theta_history']
                                ]
                              )
        self.handle[2].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['Force_history'],
                               # self.pendulums['pend_%s'%(1)]['Force_history'],
                               # self.pendulums['pend_%s'%(2)]['Force_history']
                                # self.pendulums['pend_%s'%(2)]['Force_history']
                                ]
                              )
        self.handle[3].update(self.pendulums['pend_%s'%(0)]['time_history'],
                              [self.pendulums['pend_%s'%(0)]['reward'],
                                ]
                              )

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

    def update(self, time, data):
        '''
            Adds data to the plot.
            time is a list,
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
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
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
