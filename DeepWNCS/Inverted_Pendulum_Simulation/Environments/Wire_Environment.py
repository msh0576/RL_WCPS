from Plant.pendulumSim import Pendulum
import Plant.pendulumParam as P
import numpy as np
from collections import namedtuple
from itertools import count

dtype = np.float32

THETA_ERROR_MAX = 10
THETA_ERROR_MIN = 0
THETA_MARGIN_MAX = THETA_ERROR_MAX
THETA_MARGIN_MIN = 0
AMPLITUDE = 0.1    # pendulum reference size
POSITION_ERROR_MAX = AMPLITUDE*1 + AMPLITUDE*0.2
POSITION_ERROR_MIN = 0
POSITION_MARGIN_MAX = POSITION_ERROR_MAX
POSITION_MARGIN_MIN = 0
EPISODE_DURATION_MAX = P.t_end
EPISODE_DURATION_MIN = P.t_start

class wire_environment():
    def __init__(self, network, pend_conf):
        self.network = network
        self.pend_conf = pend_conf

        self.action_space = np.array([0.], dtype=np.float32)
        self.observation_space = np.array([0., 0., 0., 0., 0.], dtype=np.float32)

    def action_spec(self):
        action_spec = namedtuple('action_spec', 'minimum, maximum, shape')
        return action_spec(-0.5, 0.5, (1,) )

    def observation_spec(self):
        observation_spec = namedtuple('observation_spec', 'shape')
        return observation_spec((5,))

    def reset(self):
        self.plant = Pendulum(self.network, self.pend_conf)
        # random initialize
        self.plant.update_command_in_plant_DDPG(0.1)
        self.plant.time_step(0)
        self.plant.update_state_in_controller()

        state = self.get_state()
        return state

    def step(self, action, t):
        '''
        Take the control command (action), and return next states of physical system
        After taking an action every 10ms, plant operates on non-action environment during the interval 9ms

            <Return>
            state: ndarray, [5,]
            reward: ndarray, [1,]
            done: bool
        '''
        if type(action) == np.ndarray:
            action = action.item(0)
        self.plant.update_command_in_plant_DDPG(action)
        self.plant.time_step(t) # just state update; apply control command to the plant
        self.plant.update_state_in_controller() # controller receives a new state of a plant
        self.run_timestep_without_action(t)

        next_state = self.get_state()
        done = self.check_termination()
        reward = self.get_reward(t, done)
        info = 0

        return next_state, reward, done, info

    def get_reward(self, t, done):
        '''
            <output>
            reward: (np.array, [1]), episode duration of a plant
        '''
        W_position = 0.6
        W_theta = 0.4
        W_duration = 0.

        error = self.get_errorVector()

        duration_reward = np.array([0.], dtype=np.float32)  # [1]
        theta_margin_reward = np.array([0.], dtype=np.float32)  # [1]
        position_margin_reward = np.array([0.], dtype=np.float32)  # [1]

        # if not done:
        #     pass
        # else:
        #     duration_reward[:] = -1

        # duration reward
        # norm_duration = round((EPISODE_DURATION_MAX - t) / (EPISODE_DURATION_MAX - EPISODE_DURATION_MIN), 3)
        # print("norm_duration:", norm_duration)
        # print("norm_duration**0.4:", norm_duration**0.4)
        # duration_reward[:] = 1 - round(norm_duration**0.4, 3)

        # theta reward
        theta_margin = THETA_ERROR_MAX - abs(error.item(1))
        norm_theta_margin = (theta_margin - THETA_MARGIN_MIN) / (THETA_MARGIN_MAX - THETA_MARGIN_MIN)
        theta_margin_reward[:] = norm_theta_margin

        # position reward
        position_margin = POSITION_ERROR_MAX - abs(error.item(0))
        norm_position_margin = (position_margin - POSITION_MARGIN_MIN) / (POSITION_MARGIN_MAX - POSITION_MARGIN_MIN)
        position_margin_reward[:] = norm_position_margin
        # print("error.item(0):{}, POSITION_ERROR_MAX:{}, norm_position_margin:{}".format(error.item(0), POSITION_ERROR_MAX, norm_position_margin))

        # print("position_margin_reward:{}, theta_margin_reward:{}, duration_reward:{}".format(position_margin_reward, theta_margin_reward, duration_reward))
        # Reward = W_position * position_margin_reward + W_theta * theta_margin_reward + W_duration * duration_reward
        Reward = W_position * position_margin_reward + W_theta * theta_margin_reward
        return Reward

    def get_errorVector(self):
        '''
        plant error means that [x(0)-ref(0); x(1)-ref(1)]
        '''
        cartPosition_ref = self.plant.currReference()
        refVector = np.array([[cartPosition_ref], [0]], dtype=dtype) # reference vector : position and theta : [2X1]
        # plant i's states
        x = self.plant.pendulum.state    # [4X1]
        plantError = np.array([[np.abs(refVector.item(0) - x.item(0))], [np.abs(refVector.item(1) - (180.0/np.pi * x.item(1)) )]], dtype=dtype)

        return plantError
    
    def Get_errorVector(self, state):
        cartPosition_ref = self.plant.currReference()
        refVector = np.array([[cartPosition_ref], [0]], dtype=dtype) # reference vector : position and theta : [2X1]
        # plant i's states
        x = state    # [4X1]
        # print("position: {}, theta: {}".format(x.item(0), (180.0/np.pi * x.item(1))))
        plantError = np.array([[np.abs(refVector.item(0) - x.item(0))], [np.abs(refVector.item(1) - (180.0/np.pi * x.item(1)) )]], dtype=dtype)

        return plantError

    def get_state(self):
        '''
        for a single plant
        output: x vector [4X1] + cart position reference [1X1] = [5X1] ->[5,]
        '''

        x = self.plant.pendulum.state    # [4X1]
        ref = np.array([[self.plant.currReference()]]) #[1X1]
        state = np.concatenate((x, ref), axis = 0)
        return state.squeeze()

    def check_termination(self):
        done = False
        error = self.get_errorVector()
        position_error = error[0][0]
        theta_error = error[1][0]
        if abs(position_error) > POSITION_ERROR_MAX:
            done = True
        elif abs(theta_error) > THETA_ERROR_MAX:
            done = True

        return done
    
    def Check_termination(self, state):
        done = False
        error = self.Get_errorVector(state)
        position_error = error[0][0]
        theta_error = error[1][0]
        # print("position_error:", POSITION_ERROR_MAX, position_error)
        # print("theta_error:", THETA_ERROR_MAX, theta_error)
        if abs(position_error) > POSITION_ERROR_MAX:
            done = True
        elif abs(theta_error) > THETA_ERROR_MAX:
            done = True

        return done


    def get_current_plant_status(self):
        r = self.plant.r
        x = self.plant.pendulum.state
        u = self.plant.prev_u

        return r, x, u

    def update_plant_state(self, t):
        self.plant.time_step(t)
        self.plant.update_state_in_controller()
    
    def get_current_env_goal(self):
        '''
            <output>
                current cart position references for all plants
                and
                pendulum theta references
        '''
        theta_ref = 0
        return self.plant.currReference(), theta_ref
    
    def close(self):
        self.reset()
    
    def Use_controller(self, t):
        self.plant.update_command_in_plant()
        self.plant.time_step(t) # just state update; apply control command to the plant
        self.plant.update_state_in_controller() # controller receives a new state of a plant

        next_state = self.get_state()
        done = self.check_termination()
        reward = self.get_reward(t, done)
        info = 0

        return next_state, reward, done, info
    
    def run_timestep_without_action(self, start_time):
        '''
        During 9ms, it runs environment steps without any action
        <Argument>
            start_time: second
        '''
        sec = round(start_time,3)

        for t in count(start=sec+0.001, step=0.001):
            if t >= sec + 0.01:
                break
            # print("time:{}s".format(round(t,3)))
            self.update_plant_state(round(t,3))
        # print("run_timestep_without_action -- {} ~ {}".format(sec, round(t,3)))
        
    