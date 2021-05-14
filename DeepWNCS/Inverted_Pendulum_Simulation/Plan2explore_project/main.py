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
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, OneStepModel
from planner import MPCPlanner
from utils import lineplot, write_video, imagine_ahead, lambda_return, compute_intrinsic_reward, FreezeParameters, ActivateParameters
from tensorboardX import SummaryWriter


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet, Dreamer or plan2explore')
parser.add_argument('--algo', type=str, default='p2e', help='planet, dreamer or p2e')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=400, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=400, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=60, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=3000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--worldmodel-MSEloss', action='store_true', help='use MSE loss for observation_model and reward_model training')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=6e-4, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=200, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
#Plan2Explore parameters
parser.add_argument('--onestep-num', type=int, default=5, metavar='H', help='numbers of onestep models')
parser.add_argument('--ensemble_loss_scale', type=float, default=1.0, metavar='H', help='weight for the ensemble loss')
parser.add_argument('--disagreement_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--onestep-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function a for a onestep dense layer')
parser.add_argument('--adaptation-step', type=int, default=1_000_000, metavar='H', help='number of step to train the actor with real task reward')
parser.add_argument('--zero-shot', action='store_true', help='use the normal actor for every test. If False, it uses curious_actor until adaptation_step')
#Dreamer parameters
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
#Planet parameters
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')

args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
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
           'onestep_loss': [], 'curious_actor_loss':[], 'curious_value_loss':[]}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
  D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
  # Initialise dataset D with S random seed episodes
  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      D.append(observation, action, reward, done)
      observation = next_observation
      t += 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)


# Initialise model parameters randomly
transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.dense_activation_function).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
if args.algo=="dreamer" or args.algo=="p2e":
  actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function).to(device=args.device)
  value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
  value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
  params_list = param_list + value_actor_param_list
if args.algo=="p2e":
  curious_actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function).to(device=args.device)
  curious_value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
  onestep_models = [OneStepModel(args.belief_size, env.action_size, args.embedding_size, args.onestep_activation_function).to(device=args.device) for _ in range(args.onestep_num)]
  onestep_param_list = []
  for x in onestep_models: onestep_param_list += list(x.parameters())
  onestep_modules = []
  for x in onestep_models: onestep_modules += x.modules
model_optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon)
if args.algo=="dreamer" or args.algo=="p2e":
  actor_optimizer = optim.Adam(actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
  value_optimizer = optim.Adam(value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)
if args.algo=="p2e":
  curious_actor_optimizer = optim.Adam(actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
  curious_value_optimizer = optim.Adam(value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)
  onestep_optimizer = optim.Adam(onestep_param_list, lr=0 if args.learning_rate_schedule != 0 else args.disagreement_learning_rate, eps=args.adam_epsilon)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  reward_model.load_state_dict(model_dicts['reward_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  model_optimizer.load_state_dict(model_dicts['model_optimizer'])
  if args.algo=="dreamer" or args.algo=="p2e":
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
  if args.algo=="p2e":
    curious_actor_model.load_state_dict(model_dicts['curious_actor_model'])
    curious_value_model.load_state_dict(model_dicts['curious_value_model'])
    onestep_optimizer.load_state_dict(model_dicts['onestep_optimizer'])
if args.algo=="dreamer":
  print("DREAMER")
  planner = actor_model
elif args.algo=="p2e":
  print("Plan2Explore") 
  planner = actor_model
  curious_planner = curious_actor_model
else:
  print("PlaNet")
  planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model)
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, device=args.device)  # Allowed deviation in KL divergence

def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False):
  # Infer belief over current state q(s_t|o≤t,a<t) from the history
  belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
  belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
  if args.algo=="dreamer" or args.algo=="p2e":
    action = planner.get_action(belief, posterior_state, det=not(explore))
  else:
    action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
  if explore:
    action = torch.clamp(Normal(action, args.action_noise).rsample(), -1, 1) # Add gaussian exploration noise on top of the sampled action
  next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
  return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
  # Set models to eval mode
  # uses normal actor for testing only case
  transition_model.eval()
  reward_model.eval()
  encoder.eval()
  with torch.no_grad():
    total_reward = 0
    for _ in tqdm(range(args.test_episodes)):
      observation = env.reset()
      belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state, action, observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
        total_reward += reward
        if args.render:
          env.render()
        if done:
          pbar.close()
          break
  print('Average Reward:', total_reward / args.test_episodes)
  env.close()
  quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []
  model_modules = transition_model.modules+encoder.modules+observation_model.modules+reward_model.modules

  print("training loop")
  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size) # Transitions start at time t = 0
    # Create initial belief and state for time t = 0
    init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:], )), nonterminals[:-1])
    # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
    if args.worldmodel_MSEloss:
      observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
    else:
      observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
      observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
    if args.algo == "p2e":
      if args.zero_shot:
        reward_dist = Normal(bottle(reward_model, (beliefs.detach(), posterior_states)),1)
      else:
        if metrics['steps'][-1]*args.action_repeat > args.adaptation_step:
          reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)),1)
        else:
          reward_dist = Normal(bottle(reward_model, (beliefs.detach(), posterior_states)),1)
      reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
    else:
      if args.worldmodel_MSEloss:
        reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))
      else:
        reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)),1)
        reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
    # transition loss
    div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
    kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
    if args.global_kl_beta != 0:
      kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0, 1))
    # Calculate latent overshooting objective for t > 0
    if args.overshooting_kl_beta != 0:
      overshooting_vars = []  # Collect variables for overshooting to process in batch
      for t in range(1, args.chunk_size - 1):
        d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
        t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
        seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
        # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
        overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
      overshooting_vars = tuple(zip(*overshooting_vars))
      # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
      beliefs, prior_states, prior_means, prior_std_devs = transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
      seq_mask = torch.cat(overshooting_vars[7], dim=1)
      # Calculate overshooting KL loss with sequence mask
      kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 
      # Calculate overshooting reward prediction loss with sequence mask
      if args.overshooting_reward_scale != 0: 
        reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 
    # Apply linearly ramping learning rate schedule
    if args.learning_rate_schedule != 0:
      for group in model_optimizer.param_groups:
        group['lr'] = min(group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule, args.model_learning_rate)
    model_loss = observation_loss + reward_loss + kl_loss
    # Update model parameters
    model_optimizer.zero_grad()
    model_loss.backward()
    nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
    model_optimizer.step()

    if args.algo=="p2e":
      #Plan2explore implementation: onestep model loss calculation and optimization
      with torch.no_grad():
        onestep_actions = actions.detach()
        onestep_obs = observations.detach()
        onestep_beliefs = beliefs.detach()
      onestep_batch_size = onestep_actions.size(1)
      action_feature_size = onestep_actions.size(2)
      obs_feature_size = args.embedding_size
      belief_feature_size = onestep_beliefs.size(2)
      with FreezeParameters(model_modules):
        onestep_embed = bottle(encoder, (onestep_obs, ))
      bagging_size = args.batch_size
      sample_with_replacement = torch.Tensor(args.onestep_num, bagging_size).uniform_(0,args.batch_size).type(torch.int64).to(device=args.device)
      for mdl in range(len(onestep_models)):
        action_indices = sample_with_replacement[mdl,:].reshape(onestep_batch_size, 1, 1).expand(onestep_batch_size, onestep_batch_size, action_feature_size)
        pred_indices = sample_with_replacement[mdl,:].reshape(onestep_batch_size, 1, 1).expand(onestep_batch_size, onestep_batch_size, obs_feature_size)
        belief_indices = sample_with_replacement[mdl,:].reshape(onestep_batch_size, 1, 1).expand(onestep_batch_size, onestep_batch_size, belief_feature_size)
        input_action = torch.gather(onestep_actions, 0, action_indices)
        input_state = torch.gather(onestep_beliefs, 0, belief_indices)
        target_prediction = torch.gather(onestep_embed, 0, pred_indices)
        prediction = onestep_models[mdl](input_state, input_action)
        prediction = prediction.mean
        loss = ((prediction - target_prediction.detach()) ** 2).mean(axis=[0,1])
        loss *= args.ensemble_loss_scale
        onestep_loss = loss.mean()
        onestep_optimizer.zero_grad()
        onestep_loss.backward()
        nn.utils.clip_grad_norm_(onestep_param_list, args.grad_clip_norm, norm_type=2)
        onestep_optimizer.step()
      
      #Plan2explore implementation: actor model loss calculation and optimization
      with torch.no_grad():
        curious_actor_states = posterior_states.detach()
        curious_actor_beliefs = beliefs.detach()
      with FreezeParameters(model_modules):
        curious_imagination_traj = imagine_ahead(curious_actor_states, curious_actor_beliefs, curious_actor_model, transition_model, args.planning_horizon)
      curious_imged_beliefs, curious_imged_prior_states, curious_imged_prior_means, curious_imged_prior_std_devs, curious_imged_actions = curious_imagination_traj
      with FreezeParameters(model_modules + value_model.modules + onestep_modules):
        curious_reward = compute_intrinsic_reward(curious_imged_beliefs, curious_imged_actions, onestep_models)
        curious_value_pred = bottle(value_model, (curious_imged_beliefs, curious_imged_prior_states))
      curious_returns = lambda_return(curious_reward, curious_value_pred, bootstrap=curious_value_pred[-1], discount=args.discount, lambda_=args.disclam)
      curious_actor_loss = -torch.mean(curious_returns)
      # Update model parameters
      curious_actor_optimizer.zero_grad()
      curious_actor_loss.backward()
      nn.utils.clip_grad_norm_(curious_actor_model.parameters(), args.grad_clip_norm, norm_type=2)
      curious_actor_optimizer.step()

      #Plan2explore implementation: curious_value model loss calculation and optimization
      with torch.no_grad():
        curious_value_beliefs = curious_imged_beliefs.detach()
        curious_value_prior_states = curious_imged_prior_states.detach()
        curious_target_return = curious_returns.detach()
      curious_value_dist = Normal(bottle(curious_value_model, (curious_value_beliefs, curious_value_prior_states)),1) # detach the input tensor from the transition network.
      curious_value_loss = -curious_value_dist.log_prob(curious_target_return).mean(dim=(0, 1)) 
      # Update model parameters
      curious_value_optimizer.zero_grad()
      curious_value_loss.backward()
      nn.utils.clip_grad_norm_(curious_value_model.parameters(), args.grad_clip_norm, norm_type=2)
      curious_value_optimizer.step()
    else:
      onestep_loss = torch.zeros(0).mean()
      curious_value_loss = torch.zeros(0).mean()
      curious_actor_loss = torch.zeros(0).mean()

    if args.algo=="p2e" and not args.zero_shot and metrics['steps'][-1]*args.action_repeat < args.adaptation_step:
      value_loss = torch.zeros(0).mean()
      actor_loss = torch.zeros(0).mean()
    else:
      if args.algo=="dreamer" or args.algo=="p2e":
        #Dreamer implementation: actor loss calculation and optimization    
        with torch.no_grad():
          actor_states = posterior_states.detach()
          actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
          imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs, imged_actions = imagination_traj
        with FreezeParameters(model_modules + value_model.modules):
          imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
          value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
        returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
        actor_loss = -torch.mean(returns)
        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()
    
        #Dreamer implementation: value loss calculation and optimization
        # print("optimize value model")
        with torch.no_grad():
          value_beliefs = imged_beliefs.detach()
          value_prior_states = imged_prior_states.detach()
          target_return = returns.detach()
        value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1)) 
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()
      else:
        value_loss = torch.zeros(0).mean()
        actor_loss = torch.zeros(0).mean()
  
  #   # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss 
            # (5) disagreement loss (6) curious_actor_loss (7) curious_value_loss
  losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item(),
                  onestep_loss.item(), curious_actor_loss.item(), curious_value_loss.item()])


  # Update and plot loss metrics
  losses = tuple(zip(*losses))
  metrics['observation_loss'].append(losses[0])
  metrics['reward_loss'].append(losses[1])
  metrics['kl_loss'].append(losses[2])
  metrics['actor_loss'].append(losses[3])
  metrics['value_loss'].append(losses[4])
  metrics['onestep_loss'].append(losses[5])
  metrics['curious_actor_loss'].append(losses[6])
  metrics['curious_value_loss'].append(losses[7])
  lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['onestep_loss']):], metrics['onestep_loss'], 'onestep_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['curious_actor_loss']):], metrics['curious_actor_loss'], 'curious_actor_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['curious_value_loss']):], metrics['curious_value_loss'], 'curious_value_loss', results_dir)


  # Data collection
  print("Data collection")
  # PlaNet, Dreamer:       Uses the reward driven planner to collect the new data.
  # Plan2Explore zeroshot: Uses the curiosity driven planner to collect the new data.
  # Plan2Explore fewshot:  Uses the curiosity driven planner until it reaches to the adaptation_step. 
  #                        After the adaptation_step, it uses the reward driven planner to collect the new data.
  if args.algo=="planet" or args.algo=="dreamer":
    policy = planner
  elif args.algo=="p2e":
    policy = curious_planner
    if not args.zero_shot and (metrics['steps'][-1]*args.action_repeat > args.adaptation_step):
      policy = planner

  with torch.no_grad():
    observation, total_reward = env.reset(), 0
    belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    for t in pbar:
      # print("step",t)
      belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, env, policy, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), explore=True)
      D.append(observation, action.cpu(), reward, done)
      total_reward += reward
      observation = next_observation
      if args.render:
        env.render()
      if done:
        pbar.close()
        break
    
    # Update and plot train reward metrics
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)


  # Test model
  print("Test model")
  if episode % args.test_interval == 0:
    # PlaNet, Dreamer:       Uses the planner that is optimized along with the world model(World model trained with data from reward driven planner).
    # Plan2Explore zeroshot: Uses the planner that is optimized along with the world model(World model trained with data from curiousity driven planner).
    # Plan2Explore fewshot:  Uses the planner that will not be trained until reaches to adaptation_step. 
    #                        After the adaptation_step it will be same as PlaNet or Dreamer
    policy = planner

    # Set models to eval mode
    transition_model.eval()
    observation_model.eval()
    reward_model.eval() 
    encoder.eval()
    if args.algo=="p2e" or args.algo=="dreamer":
      actor_model.eval()
      value_model.eval()
      if args.algo=="p2e":
        curious_actor_model.eval()
        curious_value_model.eval()
    # Initialise parallelised test environments
    test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)
    
    with torch.no_grad():
      observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
      belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, test_envs, policy, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
        total_rewards += reward.numpy()
        if not args.symbolic_env:  # Collect real vs. predicted frames for video
          video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
        observation = next_observation
        if done.sum().item() == args.test_episodes:
          pbar.close()
          break
    
    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics['test_episodes'].append(episode)
    metrics['test_rewards'].append(total_rewards.tolist())
    lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
    if not args.symbolic_env:
      episode_str = str(episode).zfill(len(str(args.episodes)))
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
    test_reward_sum = sum(metrics['test_rewards'][-1])
    writer.add_scalar("test/episode_reward", test_reward_sum/args.test_episodes, metrics['steps'][-1]*args.action_repeat)

    # Set models to train mode
    transition_model.train()
    observation_model.train()
    reward_model.train()
    encoder.train()
    if args.algo=="p2e" or args.algo=="dreamer":
      actor_model.train()
      value_model.train()
      if args.algo=="p2e":
        curious_actor_model.train()
        curious_value_model.train()
    # Close test environments
    test_envs.close()

  writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
  writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
  writer.add_scalar("observation_loss", metrics['observation_loss'][-1][0], metrics['steps'][-1])
  writer.add_scalar("reward_loss", metrics['reward_loss'][-1][0], metrics['steps'][-1])
  writer.add_scalar("kl_loss", metrics['kl_loss'][-1][0], metrics['steps'][-1])
  writer.add_scalar("actor_loss", metrics['actor_loss'][-1][0], metrics['steps'][-1])
  writer.add_scalar("value_loss", metrics['value_loss'][-1][0], metrics['steps'][-1])
  writer.add_scalar("onestep_loss", metrics['onestep_loss'][-1][0], metrics['steps'][-1]) 
  writer.add_scalar("curious_actor_loss", metrics['curious_actor_loss'][-1][0], metrics['steps'][-1]) 
  writer.add_scalar("curious_value_loss", metrics['curious_value_loss'][-1][0], metrics['steps'][-1]) 
  print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    # print("checkpoint saving model")
    torch.save({'transition_model': transition_model.state_dict(),
            'observation_model': observation_model.state_dict(),
            'reward_model': reward_model.state_dict(),
            'encoder': encoder.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            }, os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.algo=="p2e" or args.algo=="dreamer":
      # print("checkpoint saving model")
      torch.save({'actor_model': actor_model.state_dict(),
                  'value_model': value_model.state_dict(),
                  'actor_optimizer': actor_optimizer.state_dict(),
                  'value_optimizer': value_optimizer.state_dict(),
                  }, os.path.join(results_dir, 'actorvalue_models_%d.pth' % episode))
    if args.algo=="p2e":
      # print("checkpoint saving model")
      torch.save({'curious_actor_model': actor_model.state_dict(),
                  'curious_value_model': value_model.state_dict(),
                  'curious_actor_optimizer': actor_optimizer.state_dict(),
                  'curious_value_optimizer': value_optimizer.state_dict(),
                  }, os.path.join(results_dir, 'curious_models_%d.pth' % episode))
      onestep_model_dict = {'onestep_model{}'.format(i) : x.state_dict() for i,x in enumerate(onestep_models)}
      onestep_model_dict['onestep_optimizer'] = onestep_optimizer.state_dict()
      torch.save(onestep_model_dict, os.path.join(results_dir, 'onestep_models_%d.pth' % episode))

    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()