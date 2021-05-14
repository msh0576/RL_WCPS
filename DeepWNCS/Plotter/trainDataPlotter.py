# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:44:07 2020

@author: Sihoon
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def prepare_subplot(num_row, num_col):
    fig, ax = plt.subplots(num_row, num_col, sharex=True)
    return fig, ax


class trainDataPlotter:
    def __init__(self):

        self.num_rows = 3    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='Actor Losses', title='Training Data'))
        self.handle.append(myPlot(self.ax[1], ylabel='Critic Losses'))
        self.handle.append(myPlot(self.ax[2], ylabel='Episode duration'))

    def plot(self, log_data):
        self.handle[0].update(log_data['episode/count'], [log_data['episode/actor_loss']])
        self.handle[1].update(log_data['episode/count'], [log_data['episode/critic_loss']])
        self.handle[2].update(log_data['episode/count'], [log_data['episode/duration']])
        # self.handle[1].update(log_data['total/episode'], [log_data['rollout/epi_duration_history']])
        # self.handle[2].update(log_data['total/episode'], [log_data['train/loss']])

class trainDataPlotter_DQN:
    def __init__(self):

        self.num_rows = 2    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # create a handle for every subplot.
        self.handle = []
        # self.handle.append(myPlot(self.ax[0], ylabel='DQN Losses', title='Training Data'))
        self.handle.append(myPlot(self.ax[0], ylabel='Episode duration'))
        self.handle.append(myPlot(self.ax[1], ylabel='Reward mean'))

    def plot(self, log_data):
        # self.handle[0].update(log_data['episode/count'], [log_data['episode/dqn_loss']])
        self.handle[0].update(log_data['episode/count'], [log_data['episode/duration']])
        self.handle[1].update(log_data['episode/count'], [log_data['episode/reward_mean']])
        size = [i for i in range(len(log_data['episode/dqn_loss']))]

        plt.figure(num=2)
        plt.plot(size, log_data['episode/dqn_loss'])
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.show()
        print('Press key to close')
        plt.waitforbuttonpress()

class trainDataPlotter_A2C:
    def __init__(self):

        self.num_rows = 2    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # create a handle for every subplot.
        self.handle = []
        # self.handle.append(myPlot(self.ax[0], ylabel='DQN Losses', title='Training Data'))
        self.handle.append(myPlot(self.ax[0], ylabel='Episode duration'))
        self.handle.append(myPlot(self.ax[1], ylabel='Reward mean'))

    def plot(self, log_data):
        # self.handle[0].update(log_data['episode/count'], [log_data['episode/dqn_loss']])
        self.handle[0].update(log_data['episode/count'], [log_data['episode/duration']])
        self.handle[1].update(log_data['episode/count'], [log_data['episode/reward_mean']])
        size = [i for i in range(len(log_data['episode/actor_loss']))]

        fig2, ax2 = prepare_subplot(2, 1)
        handle2 = []
        handle2.append(myPlot(ax2[0], ylabel = 'actor loss'))
        handle2.append(myPlot(ax2[1], ylabel = 'critic loss'))
        handle2[0].update(size, [log_data['episode/actor_loss']])
        handle2[1].update(size, [log_data['episode/critic_loss']])

        print('Press key to close')
        plt.waitforbuttonpress()



class trainDataPlotter_DDPG:
    def __init__(self):

        self.num_rows = 2    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # create a handle for every subplot.
        self.handle = []
        # self.handle.append(myPlot(self.ax[0], ylabel='DQN Losses', title='Training Data'))
        self.handle.append(myPlot(self.ax[0], ylabel='Episode duration'))
        self.handle.append(myPlot(self.ax[1], ylabel='Reward mean'))


    def plot(self, log_data):
        # self.handle[0].update(log_data['episode/count'], [log_data['episode/dqn_loss']])
        custom_color = ['lightsteelblue', 'blue', 'lightsalmon', 'red']
        average_duration = average_log_data(log_data['episode/duration'], window = 20)
        average_reward = average_log_data(log_data['episode/reward'], window = 20)
        self.handle[0].update(log_data['episode/count'], [log_data['episode/duration'], average_duration], custom_color)
        self.handle[1].update(log_data['episode/count'], [log_data['episode/reward'], average_reward], custom_color)
        plt.pause(0.0001)
        size = [i for i in range(len(log_data['step/actor_loss']))]

        fig2, ax2 = prepare_subplot(2, 1)
        handle2 = []
        handle2.append(myPlot(ax2[0], ylabel = 'actor loss'))
        handle2.append(myPlot(ax2[1], ylabel = 'critic loss'))
        handle2[0].update(size, [log_data['step/actor_loss']])
        handle2[1].update(size, [log_data['step/critic_loss']])

        print('Press key to close')
        plt.waitforbuttonpress()


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

    def update(self, episode, data, custom_color = None):
        '''
            Adds data to the plot.
            time is a list,
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if custom_color:
            self.colors = custom_color
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(episode,
                                        data[i],
                                        color=self.colors[i%len(self.colors)],
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

def average_log_data(log_data, window = 10):
    average_log = np.zeros(len(log_data))
    for idx in range(len(log_data)):
        if idx == 0:
            average_log[0] = log_data[0]
        elif idx < window-1:
            average_log[idx] = np.mean(log_data[:idx+1])
        elif idx >= window-1:
            average_log[idx] = np.mean(log_data[idx-window+1:idx+1])
    return average_log

if __name__ == '__main__':
    test_log = [1, 4, 5, 6, 7, 8, 9, 13]
    print(" average_log:", average_log_data(test_log, window = 5))
