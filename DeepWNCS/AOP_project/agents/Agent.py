import numpy as np
import torch
import copy
import os
import pickle
import time
from itertools import count

from AOP_project.utils.perturbations import Perturber
from Environments.Wire_Environment import wire_environment
import Plant.pendulumParam as P
from Plotter.dataPlotter_v2 import dataPlotter_MPC

class Agent():
    """
    An agent class for continual lifelong learning. Handles stepping in the
    environment and logging of information. Should be subclassed for use.
    """
    def __init__(self, params):
        # Store parameters for easy access
        self.params = params

        # Environment-specific handling
        self.env = wire_environment('wire', params['pend_%s'%(0)])

        # Initialize environment
        self.time = 0
        self.epi_time = 0
        self.prev_obs = self.env.reset()

        # Perturber module allows for easy changing of dynamics
        if 'perturb_schedule' in self.params['env']:
            self.perturb = Perturber(
                self.params['env']['perturb_schedule'][0]
            )
        else:
            self.perturb = Perturber(
                {'type': 'eye', 'theta': 0, 'zero_inds': []}
            )

        # Policy testing
        self.test_pol = self.params['problem']['test_pol']
        self.eval_len = self.params['problem']['eval_len']

        # Store variables based on environment
        self.mujoco = self.params['env']['is_mujoco']
        self.has_tvel = self.params['env']['tvel']
        self.tvel = self.env.get_target_vel() if self.has_tvel else None    # target velocity?
        self.params['env']['N'] = self.params['state_dim']
        self.params['env']['M'] = self.params['action_dim']
        self.params['env']['min_act'] = self.params['min_action']
        self.params['env']['max_act'] = self.params['max_action']

        # Store important parameters for fast access
        self.T = self.params['problem']['T']    # like batch?, total?
        self.gamma = self.params['problem']['gamma']
        self.N = self.params['env']['N']
        self.M = self.params['env']['M']

        # Frozen means that no updates will occur
        self.freeze = self.params['problem']['freeze']

        # Logging history of agent/environment
        self.begin_time = 0
        self.hist = {
            'obs': np.zeros((self.T, self.N)),
            'act': np.zeros((self.T, self.M)),
            'rew': np.zeros(self.T),
            'done': np.zeros(self.T),
            'env': [[] for _ in range(self.T)],
            'total_time': np.zeros(self.T),
            'plan_time': np.zeros(self.T),
            'update_time': np.zeros(self.T),
            'env_states': [],
            'pol_test': np.zeros(self.T)
        }
        # Show real time figures
        self.has_figure = True
        if self.has_figure == True:
            self.dataPlotter = dataPlotter_MPC(self.params)

        # Default directory name to save to based on time
        if not self.params['problem']['dir_name']:
            import datetime
            now = datetime.datetime.now()
            times = (now.month, now.day, now.hour, now.minute)
            ctime = '%02d%02d_%02d%02d' % times
            self.output_dir = 'ex/' + ctime
        else:
            self.output_dir = self.params['problem']['dir_name']

        # Misc variables
        self.cache = ()
        self.dtype = torch.float32
        
        USE_GPU = self.params['problem']['use_gpu']
        self.device = torch.device('cuda' if USE_GPU and \
                                   torch.cuda.is_available() else 'cpu')

        print('Initialized %s agent' % self.params['env']['env'])
        print('Saving to dir: %s' % self.output_dir)
        print('N: %d, M: %d' % (self.N, self.M))

    def run_lifetime(self):
        """
        Run agent's lifetime; is easy to restart from previously paused agent.
        """
        fin_lifetime = False
        try:
            while self.time < self.T:   # self.time: 1 -> 10ms
                print('########## Total Time: {}0ms / Epi Time: {}0ms ############'.format(self.time, self.epi_time))
                self.run_timestep()
                
            fin_lifetime = True
        except KeyboardInterrupt:
            print('Terminating agent')
            self.env.close()
            
        self.shutdown()

        return fin_lifetime

    def run_timestep(self):
        """
        Generalized method for handling agent behaviors: updating, printing,
        stepping, etc.
        """
        # if self.params['problem']['render_env']:
        #     self.env.render()

        # if self.mujoco:
        #     self.hist['env_states'].append(
        #         copy.deepcopy(self.env.sim.get_state()))

        # Check for terminal conditions and reset if terminal
        done_signal = self.update_env()
        if done_signal and self.time > 0:
            self.hist['done'][self.time-1] = 1

        # Test the policy in standard RL setting
        # if self.test_pol:
        #     self.test_policy()

        # Save image objects for particle environments
        # if 'img_freq' in self.params['problem'] and \
        #    self.time % self.params['problem']['img_freq'] == 0 \
        #    and self.time > 0:
        #     if self.params['env']['env'] == 'Particle':
        #         self.save_particle_img_obj()
        #     elif self.params['env']['env'] == 'PWorld':
        #         self.save_pworld_img_obj()

        # if self.mujoco:
        #     env_state = self.env.sim.get_state()

        # check = time.time()
        # action = 0
        action = self.get_action(self.epi_time)
        # Add noise to action
        action += np.random.normal(
            0, self.params['problem']['act_noise'], 
            size=action.shape
        )

        # self.hist['plan_time'][self.time] += time.time() - check
        self.hist['total_time'][self.time] = self.time - self.begin_time

        # if self.mujoco:
        #     self.env.sim.set_state(env_state)

        
        self.step(action, self.epi_time)
        # Everything below self.step must refer to self.time-1 when indexing
        # into arrays

        # check = time.time()
        if not self.freeze:
            self.do_updates()
        # self.hist['update_time'][self.time-1] += time.time() - check

        # if self.time % self.params['problem']['print_freq'] == 0:
        #     self.print_logs()

        if self.time % self.params['problem']['save_freq'] == 0:
            self.save_self()
        
        

    def step(self, action, time):
        # Execute action in environment
        action = np.clip(action,
            self.params['env']['min_act'],
            self.params['env']['max_act'])
        perturbed_action = self.perturb.perturb(action)
        print("action:", perturbed_action.item())
        obs, rew, done, ifo = self.env.step(perturbed_action.item(), time*0.01)
        # obs, rew, done, ifo = self.env.Use_controller(time*0.01)  # Wired Control Test
        
        # Update history and buffers
        # if not self.params['problem']['do_resets']:
        #     done = False
        self.update_history({
            'obs': obs, 'act': action,
            'rew': rew, 'done': done
        })
        self.action_taken(self.prev_obs, obs, rew, done, ifo)
        
        # Reset if desired
        if done:
            print("episode done in step() !!!")
            self.prev_obs = self.env.reset()
            self.epi_time = 0
        else:
            self.prev_obs = obs

        # run env steps without any action: ex. from 11ms to 19ms
        self.prev_obs = run_timestep_without_action(self.env, self.epi_time)

        # Figure graph in real time
        if self.has_figure == True:
            show_figure(self.dataPlotter, self.time*0.01, self.hist['obs'][self.time], self.hist['act'][self.time], self.hist['rew'][self.time])

        # Keep track of timesteps taken in environment
        self.time += 1   # 10ms increase
        self.epi_time += 1

        return obs, rew, done, ifo

    

    def test_policy(self):
        """
        Tests the learned policy in a standard RL episode, of length eval_len.
        Subclasses should override with their method for action generation.
        """
        info = [] # eval_traj goes here
        self.hist['pol_test'][self.time] = 0

    def update_env(self):
        """
        Handles environment-specific updates that must be done. Also includes
        printing certain environment-specific metrics. If episode reset/some
        other termination method, done_signal should return True.
        """
        # if self.time % self.params['problem']['print_freq'] == 0 \
        #     and self.time > 0:
        #     self.print('environment metrics', mode='head')

        done_signal = False

        def get_ind(change_freq, schedule):
            ind = self.time // change_freq
            return ind % len(schedule)

        # Signify an episode length, if desired
        if round(self.epi_time*0.01, 3) % P.t_end == 0:
            print("episode finished!")
            self.prev_obs = self.env.reset()
            done_signal = True
            self.epi_time = 0

        # Update action perturbation module
        if 'perturb_change_every' in self.params['env'] and \
           self.time % self.params['env']['perturb_change_every'] == 0:
            ind = get_ind(
                self.params['env']['perturb_change_every'],
                self.params['env']['perturb_schedule'])
            # print("self.params['env']['perturb_schedule']:", self.params['env']['perturb_schedule'])
            # print("ind:", ind)
            perturb_params = self.params['env']['perturb_schedule'][ind]
            self.perturb = Perturber(perturb_params)

        # env_name = self.params['env']['env']
        # pf = self.params['problem']['print_freq']
        # if env_name == 'Particle':
        #     # Change the maze grid
        #     change_every = self.params['env']['pmaze_change_every']
        #     if self.time % change_every == 0:
        #         pmaze_ind = (self.time // change_every) \
        #             % len(self.params['env']['pmaze_schedule'])
        #         self.env.reset_grid(
        #             self.params['env']['pmaze_schedule'][pmaze_ind])
        #         self.world_changed()

        #     # Print distance to goal
        #     if self.time % self.params['problem']['print_freq'] == 0 \
        #         and self.time > 0:
        #         x = self.hist['obs'][self.time-1][:2]
        #         g = self.hist['obs'][self.time-1][2:]
        #         dist = np.linalg.norm(x-g)
        #         self.print('distance to goal', dist)

        # elif 'Hopper' in env_name or 'Ant' in env_name:
        #     # Update target velocity
        #     if self.params['env']['tvel'] and \
        #        self.time % self.params['env']['vel_every'] == 0:
        #         sch_ind = (self.time // self.params['env']['vel_every']) \
        #             % len(self.params['env']['vel_schedule'])
        #         self.env.set_target_vel(
        #             self.params['env']['vel_schedule'][sch_ind])
        #         self.world_changed()

        #     # Log environment information
        #     self.hist['env'][self.time].append(
        #         copy.deepcopy(self.env.sim.data.qpos))
        #     self.hist['env'][self.time].append(
        #         copy.deepcopy(self.env.sim.data.body_xpos))
 
        #     if self.has_tvel:
        #         self.hist['env'][self.time].append(
        #             self.env.get_target_vel())
        #     if 'Ant' in env_name:
        #         self.hist['env'][self.time].append(
        #             copy.deepcopy(self.env.state_vector()))

        #     # Print logging information
        #     if self.time % pf == 0 and self.time > 0:
        #         if 'Hopper' in env_name:
        #             # Accessing sim.data.qpos
        #             prev_x = self.hist['env'][self.time-pf][0][0]
        #             x = self.hist['env'][self.time][0][0]
        #             vel = (x - prev_x) / (pf * self.env.dt)
        #             z = self.hist['env'][self.time][0][1]
        #         elif 'Ant' in env_name:
        #             # Accessing state vector
        #             prev_x = self.hist['env'][self.time-pf][0][0]
        #             x = self.hist['env'][self.time][0][0]
        #             vel = (x - prev_x) / (pf * self.env.dt)
        #             z = self.hist['env'][self.time][0][2]
                    
        #         self.print('x', x)
        #         self.print('z', z)
        #         self.print('x velocity avg', vel)

        #         if self.has_tvel and \
        #             self.env.get_target_vel() is not None:
        #             self.print('target velocity', 
        #                 self.env.get_target_vel())

        #     # Update target velocity
        #     if self.has_tvel:
        #         self.tvel = self.env.get_target_vel()

        return done_signal

    def get_action(self):
        """
        General method for getting action.
        """
        raise NotImplementedError

    def do_updates(self):
        """
        General method for performing updates. Updating on a schedule must be
        handled by subclasses.
        """
        pass

    def action_taken(self, prev_obs, obs, rew, done, ifo):
        """
        General method for getting history of interaction; called after
        stepping in the true environment.
        """
        pass

    def world_changed(self):
        """
        Some (oracle) agents may want to do something at world changes.
        """
        pass

    def print(self, val_name, val=None, mode='float'):
        """
        Method for logging information.
        """
        if mode == 'head':
            print('\n %36s:' % (val_name), 12*'-')
        elif mode == 'float':
            print(' %36s: %.4f' % (val_name, val))
        else:
            print(' %36s:' % (val_name), val)

    def print_logs(self):
        """
        Prints out logging information every print_freq timesteps. Should be
        called by subclasses.
        """

        # General information
        print('=' * 60)
        print(' Timestep %d' % self.time)
        print(' %.2f sec' % (time.time() - self.begin_time))

        # Get from last time logger printed information
        bi = self.time - self.params['problem']['print_freq']
        ei = self.time

        # General agent performance metrics
        self.print('reward metrics', mode='head')

        self.print('real reward avg', 
            np.mean(self.hist['rew'][bi:ei]))
        self.print('real reward max',
            np.max(self.hist['rew'][bi:ei]))
        self.print('test policy avg',
            np.mean(self.hist['pol_test'][bi:ei]))

        if self.params['problem']['ep_len'] is None:
            run_len = 500
        else:
            run_len = self.params['problem']['ep_len']

        self.print('reward avg of last %d timesteps' % run_len,
            np.mean(self.hist['rew'][max(0,ei-run_len):ei]))
        self.print('reward max of last %d timesteps' % run_len,
            np.max(self.hist['rew'][max(0,ei-run_len):ei]))

        # Computation time metrics
        self.print('computation metrics', mode='head')

        self.print('planning seconds avg',
            np.mean(self.hist['plan_time'][bi:ei]))
        self.print('update seconds avg',
            np.mean(self.hist['update_time'][bi:ei]))

        self.print('planning seconds total',
            np.sum(self.hist['plan_time'][:ei]))
        self.print('update seconds total',
            np.sum(self.hist['update_time'][:ei]))

        return bi, ei

    def update_history(self, infos):
        """
        Updates history dictionary for logging.
        """
        for key in infos:
            if key in self.hist:
                self.hist[key][self.time] = infos[key]
            else:
                print('WARNING: %s not found in self.hist' % key)

    def save_self(self):
        """
        Saves the agent in a pickle file every save_freq timesteps.
        """
        output_file = self.output_dir + '/checkpoints'
        if not os.path.isdir(output_file):
            os.makedirs(output_file)
        file_name = output_file + '/model_' + str(self.time) + '.pkl'
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()
        print('Saved model to:', file_name)

    def shutdown(self):
        """
        Called at end of training; clean up any processes.
        """
        if self.params['problem']['render_env']:
            self.env.close()


def run_timestep_without_action(env, start_time):
    '''
    During 9ms, it runs environment steps without any action
    <Argument>
        start_time: integer: ex. start_time 1 == 10ms
    '''
    sec = round((start_time*10) * 0.001,3)    # 10ms -> 0.01s

    for t in count(start=sec+0.001, step=0.001):
        if t >= sec + 0.01:
            break
        # print("time:{}s".format(round(t,3)))
        env.update_plant_state(round(t,3))
    # print("run_timestep_without_action -- {} ~ {}".format(sec, round(t,3)))
    
    return env.get_state()


def show_figure(plotter, time, states, action, reward):
    '''
    shows a figure that presents state trajectory during simulation time
    <Argument>
        time: scalar
        states: np.ndarray, [5,1]
        reward: scalar
    '''
    plotter.update(0, time, states.item(4), states, action, reward)
    plotter.plot()