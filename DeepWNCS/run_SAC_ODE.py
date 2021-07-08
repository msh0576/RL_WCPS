from random import choices
import numpy as np
import argparse
import os
import numpy as np
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
from SAC_project.sac import SAC_upgrade, SAC_RewardModel
import itertools
from SAC_project.replay_memory import ReplayMemory, TrajectoryMemory, RewardMemory, LatentODEMemory
## for BRITS
from run_BRITS import utilize
from BRITS_project.utils import to_var, linesplot, linesubplot
import BRITS_project.models as brits_models
## for Latent_ODE
from random import SystemRandom
import latent_ode_master.lib.utils as ode_utils
from latent_ode_master.lib.plotting import *
from latent_ode_master.lib.rnn_baselines import *
from latent_ode_master.lib.ode_rnn import *
from latent_ode_master.lib.create_latent_ode_model import create_LatentODE_model
from latent_ode_master.lib.parse_datasets import parse_datasets
from latent_ode_master.lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from latent_ode_master.lib.diffeq_solver import DiffeqSolver
from latent_ode_master.mujoco_physics import HopperPhysics
from latent_ode_master.lib.utils import compute_loss_all_batches
from Dreamer_project.latentODE_func import train_latentODE, create_ODE_RNN_model

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
    parser.add_argument('--SAC_mode', type=str, choices=['SAC_origin', 'SAC_brits', 'SAC_ODE'], help='choice SAC_mode among "SAC_origin" and "SAC_brits"')
    
    ######## for BRITS ########
    parser.add_argument('--brits_model', type=str, default='rits')
    parser.add_argument('--brits_update_interval', type=int, default=100)

    ######## for Latent_ODE #########
    parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
    parser.add_argument('--niters', type=int, default=300)
    parser.add_argument('--ode_lr',  type=float, default=1e-2, help="Starting learning rate.")
    # parser.add_argument('-b', '--batch-size', type=int, default=50)
    parser.add_argument('--viz', action='store_true', help="Show plots while training")
    parser.add_argument('--save', type=str, default='latent_ode_master/experiments/', help="Path for save checkpoints")
    parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
    parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
    parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
    parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
        "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")
    parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
        "Used for periodic function demo.")
    parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
        "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
    parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
    parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
    parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
    parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
    parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

    parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

    parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

    parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
    parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

    parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
    parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

    parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
    parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

    parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
    parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

    parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
    parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

    parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
    parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
    parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")
    parser.add_argument('--latentODE_update_interval', type=int, default=500)

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
        args.device = torch.device('cuda:1')
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
def SAC_train_and_test(args, env, metrics, writer, agents, brits_args, reward_models, latentODE_args, paths):
    assert args.SAC_mode != None, "choice SAC_mode"
    model_path, results_dir = paths

    ## brits args
    brits_model, traj_seq_len, brits_optimizer, brits_results_dir = brits_args

    ## latentODE args
    latentODE_models, latentODE_optimizers, latentODE_log_path, n_traj = latentODE_args

    ## specific env observation size
    spcf_obs_size = [each_env.observation_size for each_env in env._env_list]
        
    ## Memory
    SAC_memory_list = [ReplayMemory(args.replay_size, args.seed) for _ in range(args.num_plant)]
    traj_memory = TrajectoryMemory(args.replay_size, traj_seq_len, env.observation_size, env.action_size, args.num_plant, args.seed)
    reward_memory_list = [RewardMemory(args.replay_size, args.seed) for _ in range(args.num_plant)]
    latentODE_memory_list = [LatentODEMemory(n_traj, env.observation_size, env.action_size, args.num_plant, args.seed, args.device) for _ in range(args.num_plant)]


    ## Training Loop
    total_numsteps = 0
    updates = 0
    MAEs = []
    rits_losses = []
    print("training loop start")
    for i_episode in itertools.count(1):
    # for i_episode in range(1):
        ## memory buffer init
        buff_list = [{'obs': [],
                    'action': [],
                    'next_obs': [],
                    'reward': [],
                    'mask': []} for _ in range(args.num_plant)]
        initialized_buff = [False for _ in range(args.num_plant)]
        
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        prev_rewards = [0. for _ in range(args.num_plant)]
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
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            spcf_obs_idx = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
            # print("schedule:", current_sched)   # 마지막 observed 시점에서 next_state는 안보이는 값이니까, 이 값이 ground 값과 일치해야함. 아니라면, sample 저장을 잘 못 한것.
            # print("next_state:", next_state)
            if args.SAC_mode == 'SAC_origin':
                ## memorize delayed obs
                if current_sched != args.num_plant: # if it is not a null schedule
                    memorize_delayed_obs(SAC_memory_list, buff_list, current_sched, initialized_buff, state, info['reward_list'], command_action, mask)
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
                        missedObs, esti_error = estimate_missedObs(brits_model, traj_memory, current_sched, spcf_obs_size)  # tensor(CPU), [1, 5]
                        MAEs.append(esti_error)
                        # print("esti_error:", esti_error)

                        missedReward = reward_models[current_sched].estimate_reward(missedObs.to(args.device))  # tensor(GPU), [1,]
                        memorize_esti_obs(SAC_memory_list, buff_list, current_sched, initialized_buff, missedObs, missedReward.detach().cpu(), state, command_action, mask)
                    
                    ## train brits & reward model
                    if total_numsteps % args.brits_update_interval == 0:
                        rits_loss = train_brits(brits_model, traj_memory, brits_optimizer)
                        rits_losses.append(rits_loss)
                        # traj_memory.store_dataset(batch_size=400, dataset_path=sample_path)

                        reward_model_update_flag = [True if len(reward_memory) > args.batch_size else False for reward_memory in reward_memory_list]
                        if all(reward_model_update_flag) == True:
                            reward_losses = 0
                            for reward_idx, reward_model in enumerate(reward_models):
                                reward_losses += reward_model.update_parameters(reward_memory_list[reward_idx], batch_size=args.batch_size)
                            metrics['reward_loss'].append(reward_losses)
                ### brits-end ###
            elif args.SAC_mode == 'SAC_ODE':
                if current_sched != args.num_plant:
                    latentODE_memory_list[current_sched].push(state[:, spcf_obs_idx:spcf_obs_idx+spcf_obs_size[current_sched]], \
                                                            command_action[current_sched].detach().cpu().view(1,-1), \
                                                            episode_steps)
                
                if total_numsteps > 500 and total_numsteps % args.latentODE_update_interval == 0:
                    latentODE_memory_list[0].get_latest_sample(episode_steps)

                    latentODE_train_results = train_latentODE(args, latentODE_models, latentODE_optimizers, latentODE_memory_list, batch_size=50)
                    print("latentODE_train_results['pred_y']:", latentODE_train_results['pred_y'].shape)
                    linesubplot(
                        xs = latentODE_train_results['observed_tp'],
                        ys_list1 = [latentODE_train_results['observed_data'][0, :, 0].tolist(), latentODE_train_results['observed_data'][0, :, 1].tolist(), latentODE_train_results['observed_data'][0, :, 2].tolist(), latentODE_train_results['observed_data'][0, :, 3].tolist(), latentODE_train_results['observed_data'][0, :, 4].tolist()],
                        ys_list2 = [latentODE_train_results['pred_y'][0, :, 0].tolist(), latentODE_train_results['pred_y'][0, :, 1].tolist(), latentODE_train_results['pred_y'][0, :, 2].tolist(), latentODE_train_results['pred_y'][0, :, 3].tolist(), latentODE_train_results['pred_y'][0, :, 4].tolist()],
                        legends1 = ['1-dim0', '1-dim1', '1-dim2', '1-dim3', '1-dim4'],
                        legends2 = ['2-dim0', '2-dim1', '2-dim2', '2-dim3', '2-dim4'],
                        title = 'latentODE_observed_tp',
                        subtitles = '',
                        rows = 5,
                        path=results_dir, 
                        xaxis='train_iter',
                        auto_open=True
                    )
                    # lineplot(latentODE_train_results['observed_tp'],latentODE_train_results['observed_data'], 'latentODE_observed_tp', results_dir, xaxis='train_iter')
                    lineplot(list(range(len(latentODE_train_results['loss']))), latentODE_train_results['loss'], 'latentODE_trainLoss', results_dir, xaxis='train_iter', auto_open=True)
                    lineplot(list(range(len(latentODE_train_results['mse']))), latentODE_train_results['mse'], 'latentODE_trainMSE', results_dir, xaxis='train_iter', auto_open=False)



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
                if args.SAC_mode == 'SAC_brits':
                    traj_memory.new_epi_buffer()
                elif args.SAC_mode == 'SAC_ODE':
                    [latentODE_memory.set_new_buff() for latentODE_memory in latentODE_memory_list]
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


        ## Test SAC
        if i_episode % 5 == 0 and args.eval is True:
            ## save all agent model
            for idx, agent in enumerate(agents):
                agent.save_model(model_path+"_{}".format(idx))
                reward_models[idx].save_model(model_path+"_{}_reward".format(idx))

            ## Test
            avg_reward, reward_traj = SAC_test(args, env, agents, reward_models, model_path, load_model='True')
            metrics['test_episodes'].append(i_episode)
            metrics['test_rewards'].append(round(avg_reward, 2))
            lineplot(metrics['test_episodes'][-len(metrics['test_rewards']):], metrics['test_rewards'], 'test_rewards', results_dir, xaxis='episode')
            # lineplot(metrics['steps'][-len(metrics['schedule']):], metrics['schedule'], 'schedule', results_dir, xaxis='step', mode='markers')
            linesplot(list(range(len(reward_traj['estimate']))), [reward_traj['estimate'], reward_traj['ground']], legends=['estimate', 'ground truth'], title='esti_reward_traj', path=results_dir, xaxis='step')

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


def estimate_missedObs(model, memory, schedule, spcf_obs_size):
    '''
        Input:
            model: birits
            memory: trajectory memory
            schedule: scalar

        Algo:
            it returns the fastest missing obs of the current scheduled system at the current time
        Output:
            missedObs: tensor(CPU), [1, spcf_obs_size(=5)]
    '''
    traj_data = memory.sample_batch(batch_size=1)
    traj_data = to_var(traj_data)   # to tensor
    # print("traj_data['forward']['deltas']:", traj_data['forward']['deltas'])
    ret = model.run_on_batch(traj_data, optimizer=None)

    eval_masks = ret['eval_masks'].data.cpu().numpy()
    eval_ = ret['evals'].data.cpu().numpy()
    imputation = ret['imputations'].data.cpu().numpy()

    ## compute demand observation time
    syst_obs_idx = schedule * spcf_obs_size[schedule]
    tmp = ret['eval_masks'][:, :, syst_obs_idx].squeeze().tolist()
    # print("tmp:", len(tmp))  # 무슨 의미지???
    last_idx = len(tmp)
    assert tmp[-1] == 0, "estimate_missedObs function Error"
    missedObs_idx = None
    prev_val = 0
    for idx, val in enumerate(tmp):
        if int(val) == 1 and prev_val == 0 and idx != last_idx - 1:
            missedObs_idx = idx # the missed next state at the last observed time point
        prev_val = int(val)
    assert missedObs_idx != None, "missedObs_idx Error"
    # print("schedule:", schedule)
    # print("missedObs_idx:", missedObs_idx)

    ## get missed observations
    missedObs = imputation[:, missedObs_idx, syst_obs_idx:syst_obs_idx+spcf_obs_size[schedule]]
    ground = eval_[:, missedObs_idx, syst_obs_idx:syst_obs_idx+spcf_obs_size[schedule]]

    error = np.abs(missedObs - ground)
    missedObs = torch.FloatTensor(missedObs)
    # print("eval_masks:", eval_masks)
    # print("missedObs:", missedObs)
    # print("ground:", ground)
    # print("np.where(eval_masks == 1):", np.where(eval_masks == 1))
    plot(eval_, imputation, syst_obs_idx, batch_idx=0, title='eval_traj_1to5')


    return missedObs, error.sum()

def plot(eval_, imputation, syst_obs_idx, batch_idx, title):
    eval_1 = eval_[batch_idx, :, syst_obs_idx]
    eval_2 = eval_[batch_idx, :, syst_obs_idx+1]
    eval_3 = eval_[batch_idx, :, syst_obs_idx+2]
    eval_4 = eval_[batch_idx, :, syst_obs_idx+3]
    eval_5 = eval_[batch_idx, :, syst_obs_idx+4]
    impu_1 = imputation[batch_idx, :, syst_obs_idx]
    impu_2 = imputation[batch_idx, :, syst_obs_idx+1]
    impu_3 = imputation[batch_idx, :, syst_obs_idx+2]
    impu_4 = imputation[batch_idx, :, syst_obs_idx+3]
    impu_5 = imputation[batch_idx, :, syst_obs_idx+4]
    linesubplot(xs=list(range(len(eval_1))), 
                ys_list1 = [eval_1, eval_2, eval_3, eval_4, eval_5], 
                ys_list2 = [impu_1, impu_2, impu_3, impu_4, impu_5],
                legends1 = ['eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5'],
                legends2 = ['esti_1', 'esti_2', 'esti_3', 'esti_4', 'esti_5'], 
                title = title, 
                subtitles = '',
                rows = 5,
                path=results_dir, 
                xaxis='step')

def train_brits(model, memory, brits_optimizer):
    # model.train()
    run_loss = 0.
    for idx in range(5):
        batch_data = memory.sample_batch(batch_size=64)
        batch_data = to_var(batch_data)   # to tensor
        ret = model.run_on_batch(batch_data, optimizer=brits_optimizer)

        run_loss += ret['loss'].item()
    ## for debug
    syst_obs_idx = 0
    eval_ = ret['evals'].data.cpu().numpy()
    imputation = ret['imputations'].data.cpu().numpy()
    plot(eval_, imputation, syst_obs_idx, batch_idx=0, title='eval_traj_1to5_train')
    return run_loss
    


def memorize_delayed_obs(memory_list, buff_list, current_sched, initialized_buff, state, rewards, command_action, mask):
    '''
        Input:
            current_sched: scalar
            state: tensor, [1, env_state_size]:
            rewards: list, len [# of system]: element, scalar
            command_action: tensor, [# of system, ]
        Algo:
            it pushes buff_list to the memory.
            buff_list gets the delayed obs. 
            Let schedule the system at t and t+3, then it pushes a sample (s_t, a_t, s_{t+3}, r_{t+3})
    '''
    assert isinstance(current_sched, int), 'Expected int'
    assert current_sched >= 0 and current_sched < args.num_plant, 'memory_list index error'

    spcf_obs_crite = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
    buff_list[current_sched]['next_obs'] = state[:, spcf_obs_crite:spcf_obs_crite+spcf_obs_size[current_sched]]
    buff_list[current_sched]['reward'] = torch.tensor([rewards[current_sched]], dtype=torch.float32)

    # should be: observation: tensor(CPU), [1, obs_size], action: tensor(CPU), [1, action_size(command_action)], reward: tensor(CPU), [1, ], done: tensor(CPU) of float (not bool), [1,]
    # print("before memory push --- state: {}, action: {}, reward: {}, mask: {}".format(state.shape, action.detach().cpu().shape, torch.tensor([reward], dtype=torch.float32).shape, torch.tensor([mask], dtype=torch.float32).shape))
    if initialized_buff[current_sched] == True:    # initialize check of buff_list
        memory_list[current_sched].push(buff_list[current_sched]['obs'],
                                        buff_list[current_sched]['action'], 
                                        buff_list[current_sched]['reward'], 
                                        buff_list[current_sched]['next_obs'], 
                                        torch.tensor([mask], dtype=torch.float32)) # Append transition to memory

    ## define a new sample that will be used at the next time
    buff_list[current_sched]['obs'] = buff_list[current_sched]['next_obs']
    buff_list[current_sched]['action'] = command_action[current_sched].detach().cpu().view(1,-1)
    initialized_buff[current_sched] = True


def memorize_esti_obs(memory_list, buff_list, current_sched, initialized_buff, esti_obs, esti_reward, state, command_action, mask):
    '''
        Input:
            esti_obs: tensor(CPU), [1, 5]
            esti_reward: tensor(CPU), [1,]
        Algo: 
            1. push a sample (prev_state, prev_action, esti_obs, esti_reward) to memroy
            2. define prev_state and action
        Output:
    '''
    assert isinstance(current_sched, int), 'Expected int'
    assert current_sched >= 0 and current_sched < args.num_plant, 'memory_list index error'

    if initialized_buff[current_sched] == True:
        memory_list[current_sched].push(buff_list[current_sched]['obs'],
                                        buff_list[current_sched]['action'],
                                        esti_reward,
                                        esti_obs,
                                        torch.tensor([mask], dtype=torch.float32))
    ## define a new sample
    spcf_obs_crite = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
    buff_list[current_sched]['obs'] = state[:, spcf_obs_crite:spcf_obs_crite+spcf_obs_size[current_sched]]
    buff_list[current_sched]['action'] = command_action[current_sched].detach().cpu().view(1,-1)
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

    ## BRITS model ##################################
    traj_seq_len = 80
    brits_model = getattr(brits_models, args.brits_model).Model(seq_len=traj_seq_len)
    brits_optimizer = optim.Adam(brits_model.parameters(), lr = 1e-3)
    brits_results_dir = os.path.join("BRITS_project","results", "SAC_estimate_missedObservations")
    os.makedirs(brits_results_dir, exist_ok=True)

    if torch.cuda.is_available():
        brits_model = brits_model.cuda()
    
    log_path = brits_results_dir + '/{}_log'.format(args.brits_model)
    brits_args = (brits_model, traj_seq_len, brits_optimizer, brits_results_dir)

    ## Reward model ##################################
    reward_models = [SAC_RewardModel(args, spcf_obs_size[idx], args.hidden_size, args.dense_activation_function) for idx in range(args.num_plant)]

    ## LatentODE model ###############################
    file_name = os.path.basename(__file__)[:-3]
    ode_utils.makedirs(args.save)

    latentODE_input_dim_list = [spcf_obs_size[idx] + spcf_action_size[idx] for idx in range(args.num_plant)]
    n_traj = 100
    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
    classif_per_tp = False
    n_labels = 1
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(args.device)
    z0_prior = Normal(torch.Tensor([0.0]).to(args.device), torch.Tensor([1.]).to(args.device))

    # data_obj = parse_datasets(args, args.device)
    # latentODE_input_dim_list = [data_obj["input_dim"]] * args.num_plant
    if args.ode_rnn:
        # Create ODE-GRU model
        print("ODE-RNN model!!")
        latentODE_models = [create_ODE_RNN_model(args, latentODE_input_dim_list[idx], obsrv_std, args.device, 
                            classif_per_tp = classif_per_tp, n_labels = n_labels) for idx in range(args.num_plant)]
    elif args.latent_ode:
        print("LatentODE Model!!")
        latentODE_models = [create_LatentODE_model(args, latentODE_input_dim_list[idx], z0_prior, obsrv_std, args.device, 
                        classif_per_tp = classif_per_tp,
                        n_labels = n_labels) for idx in range(args.num_plant)]
    
    latentODE_log_path = "latent_ode_master/logs/" + file_name + "_" + str(experimentID) + ".log"
    if not os.path.exists("latent_ode_master/logs/"):
        ode_utils.makedirs("latent_ode_master/logs/")
    latentODE_logger = ode_utils.get_logger(logpath=latentODE_log_path, filepath=os.path.abspath(__file__))

    latentODE_optimizers = [optim.Adamax(latentODE_models[idx].parameters(), lr=args.ode_lr) for idx in range(args.num_plant)]

    latentODE_args = (latentODE_models, latentODE_optimizers, latentODE_log_path, n_traj)

    #######################################################

    ## SAC train and test
    paths = (model_path, results_dir)
    SAC_train_and_test(args, env, metrics, writer, agent_list, brits_args, reward_models, latentODE_args, paths)

    # SAC_test(args, env, agent_list, reward_models, model_path, load_model='True')

    ## store SAC_sample 
    # make_dataset(sample_path, env, agent_list, path, args.num_plant)

    print("Finish!!!!")