# -*- coding: utf-8 -*-

'''
environments

'''
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from Models.A2C import A2C
import torch, argparse, pickle
import Plant.pendulumParam as P
from Util.utils import to_tensor, set_cuda
from Util.networkScheduler import networkScheduler
from Plotter.trainDataPlotter import trainDataPlotter_A2C
from Plotter.testDataPlotter import testDataPlotter

# hyper parameter
dtype = np.float32


class MultipendulumSim_A2C():
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.is_cuda, self.device = set_cuda()
        self.agent = A2C(conf, self.device)


    def train(self, model_path, log_path):
        self.epi_rewards_mean, self.epi_durations, self.episode = [], [], []
        self.actor_losses, self.critic_losses = [], []
        self.step_count = []
        for epi in range(self.conf['num_episode']):  # episodes
            log_probs, values, rewards, masks = [], [], [], []
            entropy, epi_reward, epi_duration, step = 0, 0.0, 0.0, 0
            print("--- episode %s ---"%(epi))
            self.episode.append(epi)

            state = self.env.reset()                    # [2*num_plant, 1]
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    # action = self.agent.select_action(state_ts)     # action type: tensor [1X1]
                    state_ts = to_tensor(state).reshape(-1)    # [1]
                    dist, value = self.agent.actor(state_ts), self.agent.critic(state_ts)   # pi(s) and V(s)
                    action = dist.sample()  # scalar of a tensor
                    next_state, reward, done, info = self.env.step(action.cpu().numpy().item(), t) # shape of next_state : [(2*num_plant) X 1]

                    log_prob = dist.log_prob(action).unsqueeze(0)   # [1] : log pi(a_t|s_t)
                    entropy += dist.entropy().mean()

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                    masks.append(torch.tensor([1-done], dtype=torch.float, device=self.device))
                    # print("step reward: ", reward)
                    state = next_state
                    epi_reward += reward
                    step += 1
                    self.step_count.append(step)
                    if done:
                        break
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts
            # optimize - monte-carlo
            actor_loss, critic_loss = self.agent.optimization_model(next_state, rewards, log_probs, values, masks)
            print("epi_duration:", epi_duration)
            print("mean epi_reward:", epi_reward/len(rewards))
            # episode done
            self.epi_rewards_mean.append(epi_reward/len(rewards))
            self.epi_durations.append(epi_duration)
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            if epi % 10 == 0:
                torch.save(self.agent.actor.state_dict(), model_path)
                self.save_log(log_path)
        # Save satet_dict
        torch.save(self.agent.actor.state_dict(), model_path)
        self.save_log(log_path)

    def train_v2(self, model_path, log_path):
        '''
        every time slot, it updates netowrk parameters like TD-error
        '''
        self.epi_rewards_mean, self.epi_durations, self.episode = [], [], []
        self.actor_losses, self.critic_losses = [], []
        self.step_count = []
        for epi in range(self.conf['num_episode']):  # episodes
            log_probs, values, rewards, masks = [], [], [], []
            entropy, epi_reward, epi_duration, step = 0, 0.0, 0.0, 0
            print("--- episode %s ---"%(epi))
            self.episode.append(epi)

            state = self.env.reset()                    # [2*num_plant, 1]
            t = P.t_start
            while t < P.t_end:        # one episode (simulation)
                epi_duration = t
                if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                    # action = self.agent.select_action(state_ts)     # action type: tensor [1X1]
                    state_ts = to_tensor(state).reshape(-1)    # [1]
                    dist, value = self.agent.actor(state_ts), self.agent.critic(state_ts)   # pi(s) and V(s)
                    action = dist.sample()  # scalar of a tensor
                    next_state, reward, done, info = self.env.step(action.cpu().numpy().item(), t) # shape of next_state : [(2*num_plant) X 1]

                    actor_loss, critic_loss = self.agent.optimization_model_v2(dist, action, state_ts, to_tensor(next_state).reshape(-1), reward, done)

                    state = next_state
                    epi_reward += reward
                    step += 1
                    self.step_count.append(step)
                    rewards.append(reward)
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
                    if done:
                        break
                else:   # every 1 ms
                    self.env.update_plant_state(t) # plant status update
                t = t + P.Ts
            # optimize
            print("epi_duration:", epi_duration)
            print("mean epi_reward:", epi_reward/len(rewards))
            # episode done
            self.epi_rewards_mean.append(epi_reward/len(rewards))
            self.epi_durations.append(epi_duration)

        # Save satet_dict
        torch.save(self.agent.actor.state_dict(), model_path)
        # torch.save(self.agent.critic.state_dict(), CRITIC_MODEL)
        self.save_log(log_path)

    def test(self, env, iteration, test_duration, model_path, algorithm = 'A2C'):
        aver_epi_duration, epi_test_actions, epi_step_count = [], [], []
        done_count = 0
        new_agent = self.select_agent(algorithm, model_path)
        t_end = test_duration

        for i in range(iteration):
            print("--- iteration %s ---"%(i))
            test_actions = []
            epi_reward, epi_duration, step_count = 0.0, 0.0, 0
            state = env.reset()
            t = P.t_start
            step = 0
            # realtimePlot = testDataPlotter(self.conf)
            while t < t_end:
                epi_duration = t
                t_next_plot = t + P.t_plot
                while t < t_next_plot:  # data plot period
                    if round(t,3)*1000 % 10 == 0: # every 10 ms, schedule udpate
                        # state_ts = to_tensor(state).reshape(-1)
                        # dist = new_agent.actor(state_ts)
                        # action = dist.sample()
                        # next_state, reward, done, info = env.step(action.cpu().numpy(), t)
                        action, next_state, reward, done, info = self.action_and_envStep(new_agent, env, t, state, algorithm)

                        state = next_state
                        epi_reward += reward
                        test_actions.append(action.item())
                        step += 1
                        step_count = step
                        if done:
                            done_count += 1
                            break
                    else:   # every 1 ms
                        env.update_plant_state(t) # plant status update
                    t = t + P.Ts
                # self.update_dataPlot(realtimePlot, t, env) # update data plot
                if done: break
            aver_epi_duration.append(epi_duration)
            epi_step_count.append(step_count)
            epi_test_actions.append(test_actions)
            print("episode duration:", epi_duration)
        print("Average episode duration: ", sum(aver_epi_duration)/iteration)
        print("unstability ratio:", done_count/iteration*100)
        realtimePlot = testDataPlotter(self.conf)
        realtimePlot.print_schedRatio(epi_test_actions, epi_step_count)
        # realtimePlot.print_totalError()


    def select_agent(self, algorithm, model_path):
        if algorithm == 'A2C':
            agent = A2C(self.conf, self.device)
            agent.actor_load_model(model_path)
        elif algorithm == 'random':
            agent = A2C(self.conf, self.device)
        elif algorithm == 'sequence':
            agent = networkScheduler(self.conf, self.device)
        return agent

    def action_and_envStep(self, agent, env, t, state, algorithm):
        if algorithm == 'A2C' or algorithm == 'random':
            state_ts = to_tensor(state).reshape(-1)
            dist = agent.actor(state_ts)
            action = dist.sample()
            # schedule = env.action_to_schedule_v2(action.cpu().numpy(), self.conf['action_dim'])
            next_state, reward, done, info = env.step(action.cpu().numpy().item(), t)
        elif algorithm == 'sequence':
            action = agent.select_seqAction()
            # schedule = env.action_to_schedule_v2(action.cpu().numpy(), self.conf['action_dim'])
            next_state, reward, done, info = env.step(action.cpu().numpy().item(), t)
            if done == True: print("done true")

        return action, next_state, reward, done, info

    def save_log(self, log_path):
        combined_stats = dict()
        combined_stats['episode/reward_mean'] = self.epi_rewards_mean
        combined_stats['step/count'] = self.step_count
        combined_stats['episode/duration'] = self.epi_durations
        combined_stats['episode/actor_loss'] = self.actor_losses
        combined_stats['episode/critic_loss'] = self.critic_losses
        combined_stats['episode/count'] = self.episode
        with open(log_path, 'wb') as f:
            pickle.dump(combined_stats,f)


    def load_log(self, log_path):
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def update_dataPlot(self, dataPlot, t, env):
        r_buff, x_buff, u_buff = env.get_current_plant_status()
        for i in range(env.num_plant): # +1 for wire network
            dataPlot.update(i, t, r_buff[i], x_buff[i], u_buff[i])
        dataPlot.plot()
        plt.pause(0.0001)

    def plot_cumulate_reward(self, log_path):
        '''
        plot log data on training results
        '''
        log_data = self.load_log(log_path)
        log_data_plotter = trainDataPlotter_A2C()
        log_data_plotter.plot(log_data)
        print('Press key to close')
        plt.waitforbuttonpress()

    def show_rewardFunction(self, figure):
        pass
