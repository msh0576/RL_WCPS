from Models.sample_env import EnvSampler
import Plant.pendulumParam as P
from itertools import count
import numpy as np


class MPC(EnvSampler):
    '''
        MPC for PETS
        action is selected according to a normal distribution
    '''
    def __init__(self, args, ensemble_model, env, agent, use_random_planner=False, use_mpc=True):
        super().__init__(env)
        self.args = args
        self.env = env
        self.agent = agent
        self.ensemble_model = ensemble_model
        self.use_random_planner = use_random_planner
        self.use_gt_dynamics = False    # ground truth dynamics or model dynamics
        self.use_mpc = use_mpc

        # Planner initialize
        if self.use_random_planner:
            self.planner = self.random_planner
        else:
            self.planner = self.cem_planner

    def train(self, state_buffer, action_buffer, reward_buffer, next_state_buffer, epochs=5):
        '''
            <Arguments>
            state_buffer: collections.deque, [buffer_size, epi_len, state_dim] :
                            where buffer_size is an accumulated episode size until now
            ...

        '''
        assert len(state_buffer) == len(action_buffer)
        input_states = [traj[:, :self.args['state_dim']] for traj in state_buffer]
        input_states = np.concatenate(input_states, axis=0) # np.ndarray, [buffer_size * epi_len, state_dim]
        assert input_states.shape[1] == self.args['state_dim']
        targets = [traj[:, :self.args['state_dim']] for traj in next_state_buffer]
        targets = np.concatenate(targets, axis=0)
        assert targets.shape[1] == self.args['state_dim']
        actions = [actions for actions in action_buffer]
        actions = np.concatenate(actions, axis=0)   # np.ndarray, [buffer_size * epi_len, action_dim=1]
        assert actions.shape[1] == self.args['action_dim']
        inputs = np.concatenate((input_states, actions), axis=1)
        assert inputs.shape[1] == (self.args['state_dim'] + self.args['action_dim'])
        self.ensemble_model.train(inputs, targets, epochs=epochs)

    def act(self, state, t):
        '''
            <Argument>
                state:
                t: time
            <Output>
                action for current state
            -----
            mu: np.ndarray, [horizon_len, ] : best action sequence during horizon
        '''

        if self.use_mpc:
            mu = self.planner(state, t)
            action = mu[:self.args['action_dim']]   # Get the first action
            action = action.copy()
            mu[:-self.args['action_dim']] = mu[self.args['action_dim']:]
            mu[-self.args['action_dim']:] = 0
            self.mu = mu
        else:
            print("need to modify self.use_mpc")
        
        return action

    def random_planner(self, state, t):
        '''
            It gives the best action sequence for a certain initial state
            action boundary : [-0.5, 0.5]
            actions: np.ndarray, [pop_size * max_iters, horizon_len * action_dim(=1)] : random action during horizon
            With the action info of each row, duplicate it by 'num_particles', and then gets average cost of the particles

            <Return>
                actions : np.ndarray, [horizon_len, ]
        '''
        # Generate M*I sequences of length T according to N(0, 0.5I)
        total_sequences = self.args['pop_size'] * self.args['max_iters']
        shape = (total_sequences, self.args['horizon_len'] * self.args['action_dim'])
        self.reset()    # resets mu and sigma
        actions = np.random.normal(self.mu, self.sigma, size=shape)
        actions = np.clip(actions, a_min=-0.5, a_max=0.5)

        if not self.use_gt_dynamics:
            repeated_actions = np.tile(actions, reps=(self.args['num_particles'], 1))   # 동일 배열 반복 ex. [pop_size * max_iters (=250), horizon_len * action_dim (=10)] -> [pop_size * max_iters * num_particles (=6000), 10]: col은 1000 * num_particles, row는 10 * "1"
            rows = repeated_actions.shape[0]
            repeated_states = np.tile(state, reps=(rows, 1))    # [pop_size * max_iters * num_particles(=1000), 5]
            costs = self.estimate_horizon_costs_with_model(repeated_states, repeated_actions, t)   # [1000, ]
            costs = costs.reshape(self.args['num_particles'], -1)   # [num_particles (=4), pop_size * max_iters (=250)]: there are M*I costs
            costs = np.mean(costs, axis=0)  # [pop_size * max_iters (=250), ]
            min_cost_idx = np.argmin(costs)
        else:
            print("need to be modified")
        return actions[min_cost_idx]



    def cem_planner(self, state, t):
        '''
            Cross entropy method optimizer. It gives the action sequence for a certain initial state
            by choosing elite sequences and using their mean.

        '''
        mu = self.mu
        sigma = self.sigma
        for i in range(self.args['max_iters']):
            # Generate M action sequences of length T according to N(mu, std)
            shape = (self.args['pop_size'], self.args['horizon_len'] * self.args['action_dim'])
            actions = np.random.normal(mu, sigma, size=shape)
            actions = np.clip(actions, a_min=-0.5, a_max=0.5)
            costs = None
            if not self.use_gt_dynamics:
                reps = (self.args['num_particles'], 1)
                repeated_actions = np.tile(actions, reps=reps)
                rows = repeated_actions.shape[0]
                states = np.tile(state, reps=(rows,1))
                costs = self.estimate_horizon_costs_with_model(states, repeated_actions, t)
                costs = costs.reshape(self.args['num_particles'], -1)
                costs = np.mean(costs, axis=0)  # there are M costs
            else:
                pass
            # Calculate mean and std using the elite action sequences
            costs = np.argsort(costs)
            elite_sequences = costs[:self.args['elite_size']]
            elite_actions = actions[elite_sequences, :]
            assert elite_actions.shape[0] == self.args['elite_size']
            mu = np.mean(elite_actions, axis=0)
            sigma = np.std(elite_actions, axis=0)
        return mu



    def estimate_horizon_costs_with_model(self, states, actions, t):
        '''
            Use the learned model to predict the next state.
            It does not need to interact with environment directly, just predict next states during the horizon.

            <Argument>
                actions: np.ndarray, [pop_size * max_iters * num_particles, horizon_len * action_dim] : repeated actions
                states: np.ndarray, [pop_size * max_iters * num_particles, state_dim] : 초기 state 
            <Output>
                cost: np.ndarray, [pop_size * max_iters * num_particles, ]: cost of the given action sequence
        '''
        rows = actions.shape[0] 
        cost = np.array([self.get_cost_duration(states[0, :], t)] * rows)
        sampler = self.ts1sampling(rows)    # [rows, horizon_len] = [1000, 10]: sample the ensemble network index
        for i in range(self.args['horizon_len']):
            idx = i * self.args['action_dim']
            action = actions[:, idx:idx + self.args['action_dim']]  # [1000, 1(action_dim)]
            idxs = sampler[:, i]
            _, _, next_states = self.ensemble_model.predict(states, action, idxs)
            states = next_states    # [rows, state_dim]
            # cost += np.apply_along_axis(self.get_cost, axis=1, arr=(states))  # [1000, ]: self.get_cost의 input (states) 를 열의 축을 따라, 즉 각 행씩 처리 (axis=1)
            cost += self.get_cost_for_array(states, t)
            # print("cost:", cost)
        return cost
    
    def get_cost_for_array(self, states, t):
        '''
            <Argument>
                states: np.ndarray, [batch, state_dim] :
                t: scalar :
            <Return>
                cost : np.ndarray, [batch, ]
        '''
        rows = states.shape[0]
        cost = np.zeros(shape=(rows))
        for i in range(rows):
            cost[i] += self.get_cost_duration(states[i], t)
        return cost

    def get_cost(self, state, t):
        '''
            *** It should be modified when we apply multiple pendulum system ***
            Cost for the current state: if cart position and pole's theta far away from its reference, the cost increases
            However, we now set it with a stable duration
        '''
        # weights for terms
        W_pos = 1
        W_theta = 2 # angle of pole

        goal_pos, goal_theta = self.env.get_current_env_goal()
        pos = state[0]
        theta = state[1]
        diff_pos = np.abs(goal_pos - pos)
        diff_theta = np.abs(goal_theta - (180.0/np.pi * theta))
        cost = W_pos * diff_pos + W_theta * diff_theta
        return cost
    
    def get_cost_duration(self, state, t):
        '''
            <Return>
                cost : scalar
        '''
        done = self.env.Check_termination(state)
        cost = 0
        if done:
            cost = 1
            # print("done")
        else:
            reward = self.env.get_reward(t, done)
            cost = 1 - reward[0]
        return cost
    
    def ts1sampling(self, rows):
        shape = (rows, self.args['horizon_len'])
        return np.random.choice(range(self.args['network_size']), size=shape, replace=True)

    def reset(self):
        '''
            Initializes variables mu and sigma

            self.mu is used to generate random action according to the distribution with 'mu' and 'sigma'
        '''
        self.mu = np.zeros(self.args['horizon_len'] * self.args['action_dim'])
        self.reset_sigma()

    def reset_sigma(self):
        '''
            Resets/initializes the value of sigma.
        '''
        sigma = [0.5 ** 0.5] * (self.args['horizon_len'] * self.args['action_dim'])
        self.sigma = np.array(sigma)
    
    def get_plant_status(self):
        return self.env.get_current_plant_status()

class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return np.random.uniform(low=-0.5, high=0.5, size=self.action_dim)