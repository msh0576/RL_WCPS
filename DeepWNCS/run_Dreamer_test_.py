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
from Dreamer_project.models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel
from Dreamer_project.planner import MPCPlanner
from Dreamer_project.utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, ActivateParameters
from Models.model import Actor_DDPG_v2, Critic_DDPG


from tensorboardX import SummaryWriter
import time

# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--num_plant', type=int, default=1, help='number of plants')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + MY_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
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
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + '[Options]')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))



# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  print("using CUDA")
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  print("using CPU")
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': [],
           'position': [], 'velocity': [], 'pole_xpos': [], 'pole_ypos': [], 'schedule': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialize training environment and test random actions
env = TOTAL_ENV(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.num_plant, args.render)
observation, done, t = env.reset(), False, 0

while not done:
  action = env.sample_random_action()
  schedule, _ = action
  next_observation, reward, done = env.step(action)   # 10ms step
  observation = next_observation
  metrics['position'].append(observation[0][0].item())
  metrics['velocity'].append(observation[0][3].item())
  metrics['pole_xpos'].append(observation[0][2].item())
  metrics['pole_ypos'].append(observation[0][1].item())
  metrics['schedule'].append(schedule.item())
  metrics['steps'].append(t * args.action_repeat)
  t += 1
  env.render()
env.close()

# plot observations
lineplot(metrics['steps'][-len(metrics['position']):], metrics['position'], 'cart position', results_dir, xaxis='step')
lineplot(metrics['steps'][-len(metrics['velocity']):], metrics['velocity'], 'cart velocity', results_dir, xaxis='step')
lineplot(metrics['steps'][-len(metrics['pole_xpos']):], metrics['pole_xpos'], 'pole xpos', results_dir, xaxis='step')
lineplot(metrics['steps'][-len(metrics['pole_ypos']):], metrics['pole_ypos'], 'pole ypos', results_dir, xaxis='step')
lineplot(metrics['steps'][-len(metrics['schedule']):], metrics['schedule'], 'schedule', results_dir, xaxis='step', mode='markers')

# DDPG algorithm 
DDPG_actor_model = Actor_DDPG_v2(env.observation_size, env.action_size).to(device=args.device)
DDPG_actor_target = Actor_DDPG_v2(env.observation_size, env.action_size).to(device=args.device)
DDPG_actor_target.load_state_dict(DDPG_actor_model.state_dict())
DDPG_actor_optim = optim.Adam(DDPG_actor_model.parameters(), lr=1e-4)

DDPG_critic_model = Critic_DDPG(env.observation_size, env.action_size).to(device=args.device)
DDPG_critic_target = Critic_DDPG(env.observation_size, env.action_size).to(device=args.device)
DDPG_critic_target.load_state_dict(DDPG_critic_model.state_dict())
DDPG_critic_optim = optim.Adam(DDPG_critic_model.parameters(), lr=1e-4)

# Training
for episode in tqdm(range(args.episodes + 1)):

