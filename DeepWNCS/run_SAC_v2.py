from random import choices
import numpy as np
import argparse
import os
import copy as cp

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from Dreamer_project.env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher, MY_ENVS
from Dreamer_project.total_env import TOTAL_ENV
from Dreamer_project.memory import ExperienceReplay
from Dreamer_project.models import bottle, Encoder, ObservationModel, TransitionModel, ValueModel, ActorModel
from Dreamer_project.planner import MPCPlanner
from Dreamer_project.utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, ActivateParameters, sequentialSchedule
from Models.model import Actor_DDPG_v2, Critic_DDPG, ReplayMemory_test
from Models.DDPG import DDPG_test
from Util.utils import to_tensor, make_dataset
from collections import Counter
from operator import itemgetter

## for SAC
from SAC_project.sac import SAC_upgrade, SAC_RewardModel, SAC_britsModel
import itertools
from SAC_project.replay_memory import ReplayMemory, TrajectoryMemory, RewardMemory
## for BRITS
from run_BRITS import utilize
from BRITS_project.utils import to_var, linesplot, linesubplot
import BRITS_project.models as brits_models
from Dreamer_project.brits_func import memorize_esti_obs, memorize_delayed_info, to_state_cash, check_do_estimate


from tensorboardX import SummaryWriter
import time

sample_path = './BRITS_project/json' + '/sac_dataset_randomTraj.pickle'

def hyperparameters():
    # Hyperparameters
    parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
    parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--num_plant', type=int, default=1, help='number of plants')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + MY_ENVS, help='Gym/Control Suite environment')
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length') # episode length = 10s = 1000 time step
    parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
    parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
    parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
    parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
    parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
    parser.add_argument('--worldmodel-LogProbLoss', action='store_true', help='use LogProb loss for observation_model and reward_model training')
    parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
    parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
    parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
    parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
    parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
    parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
    parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
    # Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
    parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
    parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
    parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
    parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
    parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
    parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
    parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')

    ###### for SAC #########
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--SAC_mode', type=str, choices=['SAC_origin', 'SAC_brits'], help='choice SAC_mode among "SAC_origin" and "SAC_brits"')
    
    ######## for BRITS ########
    parser.add_argument('--brits_model', type=str, default='rits')
    parser.add_argument('--brits_update_interval', type=int, default=100)

    args = parser.parse_args()
    print("here ok")
    args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
    print(' ' * 26 + '[Options]')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))
    
    return args

def setup(args):
    # Setup
    results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.disable_cuda:
        print("using CUDA")
        # args.device = torch.device('cuda:1')
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        print("using CPU")
        args.device = torch.device('cpu')
    metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 
            'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': [],
            'position': [], 'velocity': [], 'pole_xpos': [], 'pole_ypos': [], 'schedule': [],}

    summary_name = results_dir + "/{}_{}_log"
    writer = SummaryWriter(summary_name.format(args.env, args.id))

    return results_dir, metrics, writer

# Initialize training environment and test random actions
def test_random_actions(env, metrics):
    observation, done, t = env.reset(), False, 0

    while not done:
        action = env.sample_random_action()
        # print("initla action:", action)
        next_observation, reward, done, info = env.step(action, action_repeat_render=False)   # 10ms step
        observation = next_observation
        # metrics['position'].append(observation[0][0].item())
        # metrics['velocity'].append(observation[0][3].item())
        # metrics['pole_xpos'].append(observation[0][2].item())
        # metrics['pole_ypos'].append(observation[0][1].item())
        # metrics['schedule'].append(info['schedule'])
        # metrics['steps'].append(t * args.action_repeat)
        t += 1
        # env.render()
    env.close()
    # plot observations
    # lineplot(metrics['steps'][-len(metrics['position']):], metrics['position'], 'cart position', results_dir, xaxis='step')
    # lineplot(metrics['steps'][-len(metrics['velocity']):], metrics['velocity'], 'cart velocity', results_dir, xaxis='step')
    # lineplot(metrics['steps'][-len(metrics['pole_xpos']):], metrics['pole_xpos'], 'pole xpos', results_dir, xaxis='step')
    # lineplot(metrics['steps'][-len(metrics['pole_ypos']):], metrics['pole_ypos'], 'pole ypos', results_dir, xaxis='step')
    # lineplot(metrics['steps'][-len(metrics['schedule']):], metrics['schedule'], 'schedule', results_dir, xaxis='step', mode='markers')

##### DDPG algorithm #####
def DDPG_dummy_func():
    schedule_size = args.num_plant + 1
    DDPG_agent = DDPG_test(env.observation_size, env.action_size, schedule_size, args.device)
    memory = ReplayMemory_test(args.experience_size)

    ### Training
    for episode in tqdm(range(args.episodes + 1)):
        observation, total_reward = env.reset(), 0  # tensor(cpu)

        # pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        epi_reward = 0
        for t in range(args.max_episode_length // args.action_repeat):
            action = DDPG_agent.get_action(observation.to(device=args.device))   # tensor(GPU)
            next_observation, reward, done, info = env.step(action)   # 10ms step

            # should be observation: tensor(CPU), [1, obs_size], action: tensor(CPU), [1, action_size], reward: tensor(CPU), [1, ], done: tensor(CPU) of float (not bool), [1,]
            memory.append(observation, action.view(1, env.action_size).detach().cpu(), next_observation,\
                to_tensor(np.asarray(reward).reshape(1)), to_tensor(np.asarray(float(done)).reshape(1)))   
            epi_reward += reward
            observation = next_observation
        critic_loss, actor_loss = DDPG_agent.update_policy(memory, args.batch_size)

        # logging
        metrics['episodes'].append(episode)
        metrics['actor_loss'].append(actor_loss)
        metrics['value_loss'].append(critic_loss)
        metrics['train_rewards'].append(epi_reward)

        ### Test model
        if episode % args.test_interval == 0:
            print("Test model!")
            DDPG_agent.set_eval_mode()

            # Initialize parallelized test environments
            test_envs = TOTAL_ENV(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.num_plant, action_repeat_render=True)

            with torch.no_grad():
                observation, total_rewards = test_envs.reset(), 0
                pbar = tqdm(range(args.max_episode_length // args.action_repeat))
                for t in pbar:
                    action = DDPG_agent.get_action(observation.to(device=args.device))
                    next_observation, reward, done, info = test_envs.step(action)

                    total_rewards += reward
                    observation = next_observation

                    if done == True:
                        break
            # update and plot reward metrics
            metrics['test_episodes'].append(episode)
            metrics['test_rewards'].append(total_rewards)
            lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

            # set models to train
            DDPG_agent.set_train_mode()
            test_envs.close()
        
        ### Checkpoint models
        if episode % args.checkpoint_interval == 0:
            torch.save({'actor_model': DDPG_agent.actor.state_dict(),
                        'actor_target_model': DDPG_agent.actor_target.state_dict(),
                        'actor_optimizer': DDPG_agent.actor_optim.state_dict(),
                        'critic_model': DDPG_agent.critic.state_dict(),
                        'critic_target_model': DDPG_agent.critic_target.state_dict(),
                        'critic_optimizer': DDPG_agent.critic_optim.state_dict()
                        }, os.path.join(results_dir, 'models_%d.pth' % episode))

    lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir, xaxis='step')
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir, xaxis='step')
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir, xaxis='step')

##### SAC algorithm #####
def SAC_train_and_test(args, env, metrics, writer, agents, brits_args, reward_models, paths):
    assert args.SAC_mode != None, "choice SAC_mode"
    model_path, results_dir = paths

    ## brits args
    brits_model, traj_seq_len, brits_results_dir = brits_args

    ## specific env observation size
    spcf_obs_size = [each_env.observation_size for each_env in env._env_list]
        
    ## Memory
    SAC_memory_list = [ReplayMemory(args.replay_size, args.seed) for _ in range(args.num_plant)]
    traj_memory = TrajectoryMemory(args.replay_size, traj_seq_len, env.observation_size, env.action_size, args.num_plant, args.seed)
    reward_memory_list = [RewardMemory(args.replay_size, args.seed) for _ in range(args.num_plant)]

    ## Training Loop
    total_numsteps = 0
    updates = 0
    MAEs = []
    rits_losses = []
    rits_eval_losses = []
    print("training loop start")
    for i_episode in itertools.count(1):
    # for i_episode in range(1):
        ## memory buffer init
        state_cash = [{'time_step': 0,
                    'obs': torch.zeros(1, spcf_obs_size[idx_]),
                    'action': torch.zeros(1, 1),
                    'next_obs': [],
                    'reward': [],
                    'mask': []} for idx_ in range(args.num_plant)]
        initialized_buff = [False for _ in range(args.num_plant)]
        
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        prev_rewards = [0. for _ in range(args.num_plant)]
        after_train_flag = False
        while not done:
            # print("total_numsteps:",total_numsteps)
            # print("episode_steps:", episode_steps)
            if args.start_steps > total_numsteps:
                action = env.sample_random_action()  # Sample random action
            else:
                action = env.select_actions(state, agent_list)
            
            # print("here state shape:", state.shape)    # tensor(CPU), [1, state_size]
            ## Environment step
            next_state, reward, done, info = env.step(action)

            ## after processes variables of environment interaction
            current_sched = int(info['schedule'])
            command_action = info['command_actions']    # only command_actions are buffered in a replay memory
            episode_reward += reward
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            spcf_obs_idx = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
            # print("------------------------")
            # print("At {} step, schedule:{}, next_state: {}".format(episode_steps, current_sched, next_state))
            if args.SAC_mode == 'SAC_origin':
                ## memorize delayed obs
                if current_sched != args.num_plant: # if it is not a null schedule
                    memorize_delayed_obs(SAC_memory_list, state_cash, current_sched, initialized_buff, state, info['reward_list'], command_action, mask)
            elif args.SAC_mode == 'SAC_brits':
                ### brits ###
                ## memorize reward sample
                if current_sched != args.num_plant:
                    reward_memory_list[current_sched].push(torch.tensor([prev_rewards[current_sched]], dtype=torch.float32), state[:, spcf_obs_idx:spcf_obs_idx+spcf_obs_size[current_sched]])

                ## memorize trajectory info
                traj_memory.push_sample(state, current_sched, command_action)
                if len(traj_memory) > traj_seq_len * 5:
                    ## estimates missed observations and reward
                    if current_sched != args.num_plant and episode_steps > traj_seq_len: # if it is not a null schedule
                        if check_do_estimate(state_cash[current_sched]['time_step'], episode_steps) == False:
                            SAC_memory_list[current_sched].push(
                                state_cash[current_sched]['obs'], state_cash[current_sched]['action'], 
                                torch.FloatTensor([info['reward_list'][current_sched]]), state[:, spcf_obs_idx:spcf_obs_idx+spcf_obs_size[current_sched]], torch.FloatTensor([mask])
                            )
                        else:
                            rits_eval_loss, esti_obs, esti_error, ground = brits_model.estimate_missedObs(traj_memory, current_sched, spcf_obs_size)  # tensor(CPU), [1, 5]
                            MAEs.append(esti_error)
                            rits_eval_losses.append(rits_eval_loss)
    
                            esti_reward = reward_models[current_sched].estimate_reward(esti_obs.to(args.device))  # tensor(GPU), [1,]

                            # print("At {} step, schedule:{}, ground:{}, missedObs:{}".format(episode_steps, current_sched, ground, esti_obs))
                            SAC_memory_list[current_sched].push(
                                state_cash[current_sched]['obs'], state_cash[current_sched]['action'], 
                                esti_reward.detach().cpu(), esti_obs, torch.FloatTensor([mask])
                            )
                            # memorize_delayed_info(SAC_memory_list[current_sched], state_cash[current_sched], 
                            #                     state[:, spcf_obs_idx:spcf_obs_idx+spcf_obs_size[current_sched]], 
                            #                     command_action[current_sched].detach().cpu().view(1,-1),
                            #                     esti_reward.detach().cpu(),
                            #                     esti_obs,
                            #                     mask)
                                            
                    ## train brits & reward model
                    if total_numsteps % args.brits_update_interval == 0:
                        rits_loss = brits_model.train(traj_memory)
                        rits_losses.append(rits_loss)
                        # traj_memory.store_dataset(batch_size=400, dataset_path=sample_path)

                        reward_model_update_flag = [True if len(reward_memory) > args.batch_size else False for reward_memory in reward_memory_list]
                        if all(reward_model_update_flag) == True:
                            reward_losses = 0
                            for reward_idx, reward_model in enumerate(reward_models):
                                reward_losses += reward_model.update_parameters(reward_memory_list[reward_idx], batch_size=args.batch_size)
                            metrics['reward_loss'].append(reward_losses)
                        
                ### brits-end ###
            if current_sched != args.num_plant:
                to_state_cash(state_cash[current_sched], episode_steps, state[:, spcf_obs_idx:spcf_obs_idx+spcf_obs_size[current_sched]], command_action[current_sched].detach().cpu().view(1,-1), mask)
            episode_steps += 1
            total_numsteps += 1
            state = next_state
            prev_rewards = info['reward_list']

            ## update SAC models
            memory_length_flag = [True if len(memory)>args.batch_size else False for memory in SAC_memory_list]
            if all(memory_length_flag) == True:
                ## Number of updates per step in environment
                for agent_idx in range(len(agents)):
                    updates = 0
                    for i in range(args.updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agents[agent_idx].update_parameters(SAC_memory_list[agent_idx], args.batch_size, updates)
                        updates += 1
            if done == True:
                traj_memory.new_epi_buffer()
                print("episode done!")
        env.close()
        if total_numsteps > args.num_steps:
            print("break")
            break
        
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        
        metrics['episodes'].append(i_episode)
        metrics['train_rewards'].append(episode_reward)

        lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir, xaxis='episode')

        if args.SAC_mode == 'SAC_brits':
            lineplot(list(range(len(metrics['reward_loss']))), metrics['reward_loss'], 'reward_loss', results_dir, xaxis='step')
            lineplot(list(range(len(MAEs))), MAEs, 'error of missedObs estimation', results_dir, xaxis='step')
            lineplot(list(range(len(rits_losses))), rits_losses, title='rits_losses', path=results_dir, xaxis='epoch')
            lineplot(list(range(len(rits_eval_losses))), rits_eval_losses, title='rits_eval_losses', path=results_dir, xaxis='step')


        ## Test SAC
        if i_episode % 5 == 0 and args.eval is True:
            ## save all agent model
            for idx, agent in enumerate(agents):
                agent.save_model(model_path+"_{}".format(idx))
                reward_models[idx].save_model(model_path+"_{}_reward".format(idx))

            ## Test
            avg_reward, reward_traj = SAC_test(args, env, agents, reward_models, model_path, load_model='True')
            # metrics['test_episodes'].append(i_episode)
            # metrics['test_rewards'].append(round(avg_reward, 2))
            # lineplot(metrics['test_episodes'][-len(metrics['test_rewards']):], metrics['test_rewards'], 'test_rewards', results_dir, xaxis='episode')
            # linesplot(list(range(len(reward_traj['estimate']))), [reward_traj['estimate'], reward_traj['ground']], legends=['estimate', 'ground truth'], title='esti_reward_traj', path=results_dir, xaxis='step')

            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
    env.close()

def SAC_test(args, env, agents, reward_models, path, load_model='False'):
    if load_model == 'True':
        for idx, agent in enumerate(agents):
            agent.load_model(path+"_{}".format(idx), evaluate='True')
            reward_models[idx].load_model(path+"_{}_reward".format(idx))

    reward_traj ={
                'ground': [],
                'estimate': []
                }
    estimate_rewards = 0

    avg_reward, episodes, t = 0., 10, 0
    test_render = False
    for idx in range(episodes):
        if idx == episodes-1:
            test_render = True
            schedules = []
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # action = agent.select_action(state, evaluate=True)
            action = env.select_actions(state, agents, evaluate=True)

            next_state, reward, done, info = env.step(action, test_render)
            episode_reward += reward

            if idx == episodes-1:
                schedules.append(info['schedule'])
            state = next_state
            t += 1

            ## reward model test
            if args.SAC_mode == 'SAC_brits':
                spcf_obs_size = 5
                next_state = torch.FloatTensor(next_state).to(args.device)
                esti_rewards = 0
                for reward_idx, reward_model in enumerate(reward_models):
                    spcf_next_state = next_state[:, (reward_idx * spcf_obs_size):(reward_idx * spcf_obs_size)+spcf_obs_size]
                    esti_rewards += reward_model.estimate_reward(spcf_next_state).item()
                reward_traj['ground'].append(reward)
                reward_traj['estimate'].append(esti_rewards)
        avg_reward += episode_reward
    avg_reward /= episodes
    schedules = Counter(schedules)
    schedule_ratio = sorted(schedules.items(), key=itemgetter(0))

    env.close()

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("schedule_ratio:", schedule_ratio)
    if args.SAC_mode == 'SAC_brits':
        esti_avg_reward = sum(reward_traj['estimate'])/episodes
        print("avg_reward:{}, esti_avg_reward:{}".format(avg_reward, esti_avg_reward))
    print("----------------------------------------")
    return avg_reward, reward_traj


def memorize_delayed_obs(memory_list, state_cash, current_sched, initialized_buff, state, rewards, command_action, mask):
    '''
        Input:
            current_sched: scalar
            state: tensor, [1, env_state_size]:
            rewards: list, len [# of system]: element, scalar
            command_action: tensor, [# of system, ]
        Algo:
            it pushes state_cash to the memory.
            state_cash gets the delayed obs. 
            Let schedule the system at t and t+3, then it pushes a sample (s_t, a_t, s_{t+3}, r_{t+3})
    '''
    assert isinstance(current_sched, int), 'Expected int'
    assert current_sched >= 0 and current_sched < args.num_plant, 'memory_list index error'

    spcf_obs_crite = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
    state_cash[current_sched]['next_obs'] = state[:, spcf_obs_crite:spcf_obs_crite+spcf_obs_size[current_sched]]
    state_cash[current_sched]['reward'] = torch.tensor([rewards[current_sched]], dtype=torch.float32)

    # should be: observation: tensor(CPU), [1, obs_size], action: tensor(CPU), [1, action_size(command_action)], reward: tensor(CPU), [1, ], done: tensor(CPU) of float (not bool), [1,]
    # print("before memory push --- state: {}, action: {}, reward: {}, mask: {}".format(state.shape, action.detach().cpu().shape, torch.tensor([reward], dtype=torch.float32).shape, torch.tensor([mask], dtype=torch.float32).shape))
    if initialized_buff[current_sched] == True:    # initialize check of state_cash
        memory_list[current_sched].push(state_cash[current_sched]['obs'],
                                        state_cash[current_sched]['action'], 
                                        state_cash[current_sched]['reward'], 
                                        state_cash[current_sched]['next_obs'], 
                                        torch.tensor([mask], dtype=torch.float32)) # Append transition to memory

    ## define a new sample that will be used at the next time
    state_cash[current_sched]['obs'] = state_cash[current_sched]['next_obs']
    state_cash[current_sched]['action'] = command_action[current_sched].detach().cpu().view(1,-1)
    initialized_buff[current_sched] = True



if __name__ == "__main__":
    args = hyperparameters()
    results_dir, metrics, writer = setup(args)

    ## Initialize training environment
    env = TOTAL_ENV(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.num_plant, 'random')
    print("env.observation_size, env.action_size:", env.observation_size, env.action_size)
    # D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
    ## specific env observation size
    spcf_obs_size = [each_env.observation_size for each_env in env._env_list]
    spcf_action_size = [each_env.action_size for each_env in env._env_list]
    print("spcf_obs_size:", spcf_obs_size)

    # n systems agents
    agent_list = [SAC_upgrade(spcf_obs_size[idx], spcf_action_size[idx], args) for idx in range(args.num_plant)]
    # agent = SAC_upgrade(env.observation_size, env.action_size, args)   # observation_size, action_size: 10 (5 * num_plant), 2 (1 * num_palnt)

    # saving models path
    model_path = "./saving_models/sac_model"

    ## BRITS model
    traj_seq_len = 80
    brits_model = SAC_britsModel(args, traj_seq_len, spcf_obs_size, spcf_action_size)
    brits_results_dir = os.path.join("BRITS_project","results", "SAC_estimate_missedObservations")
    os.makedirs(brits_results_dir, exist_ok=True)

    log_path = brits_results_dir + '/{}_log'.format(args.brits_model)
    brits_args = (brits_model, traj_seq_len, brits_results_dir)

    ## Reward model
    reward_models = [SAC_RewardModel(args, spcf_obs_size[idx], args.hidden_size, args.dense_activation_function) for idx in range(args.num_plant)]

    ## test random actions
    # test_random_actions(env, metrics)

    ## SAC train and test
    paths = (model_path, results_dir)
    SAC_train_and_test(args, env, metrics, writer, agent_list, brits_args, reward_models, paths)

    # SAC_test(args, env, agent_list, reward_models, model_path, load_model='True')

    ## store SAC_sample 
    # make_dataset(sample_path, env, agent_list, path, args.num_plant)

    print("Finish!!!!")