import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from SAC_project.utils import soft_update, hard_update
from SAC_project.model import GaussianPolicy, QNetwork, DeterministicPolicy
from Dreamer_project.utils import sequentialSchedule

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


### By sihoon
class SAC_upgrade(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        self.critic = QNetwork(num_inputs, action_space, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.num_plant = args.num_plant

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                print("automatic_entropy_tuning")
                self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            #$# (origin sign) self.policy = GaussianPolicy(num_inputs, action_space, args.hidden_size, action_space).to(self.device)
            self.policy = GaussianPolicy(num_inputs, action_space, args.hidden_size, self.device).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_actions(self, state, evaluate=False):
        '''
            Args:
                state: tensor(CPU), [1, (observation_size * num_plant)]
            Returns:
                output should be: tensor(CPU), [1, action_size (schedule_size + command_action_size)]
        '''
        state = torch.FloatTensor(state).to(self.device)    # tensor(GPU), [1, 5]
        schedule = sequentialSchedule(self.num_plant + 1)       # tensor(CPU), [1, 1 (schedule)]
        
        if evaluate is False:
            command_action, _, _ = self.policy.sample_v2(state)     # tensor(GPU), [1, command_size]
        else:
            _, _, command_action = self.policy.sample_v2(state)
        # print("schedule shape:", schedule.shape)
        # print("command_action shape:", command_action.shape)
        action = torch.cat([schedule.view(1,1), command_action.detach().cpu()], dim = 1)
        return action
    
    def select_action(self, state, evaluate=False):
        '''
            Inputs:
                state: tensor(CPU), [1, (observation_size)]

            Outputs:
                action should be: tensor(CPU), [1, command_action_size(1)]
        '''
        state = torch.FloatTensor(state).to(self.device)    # tensor(GPU), [1, 5]
        # print("state:", state)
        if evaluate is False:
            command_action, _, _ = self.policy.sample_v2(state)     # tensor(GPU), [1, command_size]
        else:
            _, _, command_action = self.policy.sample_v2(state)
        # print("command_action:", command_action)    
        return command_action.detach().cpu()

    def update_parameters(self, memory, batch_size, updates):
        # print("----update parameters----")

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        # memory.sample outputs : 
        # state_batch {tensor(CPU) [batch, 1, state_size]}, action_batch {torch(CPU), [batch, 1, action_size]}, reward_batch {torch(CPU), [batch, 1]}
        # print("state_batch: {}, action_batch: {}, reward_batch: {}, mask_batch: {}".format(state_batch.shape, action_batch.shape, reward_batch.shape, mask_batch.shape))

        state_batch = torch.FloatTensor(state_batch).to(self.device).squeeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device).squeeze(1)  # [50, 5]
        action_batch = torch.FloatTensor(action_batch).to(self.device).squeeze(1)   # [50, 1]
        reward_batch = torch.FloatTensor(reward_batch).to(self.device) # [50, 1]
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        # print("next_state_batch shape:", next_state_batch.shape)
        # print("action_batch shape:", action_batch.shape)
        # print("reward_batch shape:", reward_batch.shape)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample_v2(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample_v2(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        # if updates % self.target_update_interval == 0:
        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists('saving_models/'):
            os.makedirs('saving_models/')

        # if actor_path is None:
        #     actor_path = "saving_models/sac_actor_{}_{}".format(env_name, suffix)
        # if critic_path is None:
        #     critic_path = "saving_models/sac_critic_{}_{}".format(env_name, suffix)
        

        print('Saving models to {}'.format(path))
        # torch.save(self.policy.state_dict(), actor_path)
        # torch.save(self.critic.state_dict(), critic_path)

        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        }, path)

    # Load model parameters
    def load_model(self, path, evaluate=False):
        print('Loading models from {}'.format(path))
        # if actor_path is not None:
        #     self.policy.load_state_dict(torch.load(actor_path))
        #     self.policy.eval()
        # if critic_path is not None:
        #     self.critic.load_state_dict(torch.load(critic_path))
        #     self.critic.eval()

        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])

        if evaluate == True:
            self.policy.eval()
            self.critic.eval()
        else:
            self.policy.train()
            self.critic.train()

from SAC_project.model import RewardModel
class SAC_RewardModel(object):
    def __init__(self, args, state_size, hidden_size, activation_function='relu'):
        self.reward_model = RewardModel(state_size, hidden_size, activation_function).to(args.device)
        self.optimizer = Adam(self.reward_model.parameters(), lr=args.lr)
        self.device = args.device
        self.cnt = 0
    
    def estimate_reward(self, state):
        '''
            Input:
                state: tensor(GPU), [1, spcf_obs_size(5)] : 
            Ouput:
                missedReward: tensor(GPU), [1,]
        '''
        return self.reward_model(state)
    
    def update_parameters(self, memory, batch_size):
        reward_batch, next_state_batch = memory.sample(batch_size=batch_size)
        # memory.sample outputs : 
        # state_batch {tensor(CPU) [batch, 1, state_size]}, action_batch {torch(CPU), [batch, 1, action_size]}, reward_batch {torch(CPU), [batch, 1]}
        # print("state_batch: {}, action_batch: {}, reward_batch: {}, mask_batch: {}".format(state_batch.shape, action_batch.shape, reward_batch.shape, mask_batch.shape))

        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device).squeeze(1)  # [50, 5]
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).squeeze(1) # [50, ]
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        esti_reward = self.estimate_reward(next_state_batch)
        reward_loss = F.mse_loss(esti_reward, reward_batch, reduction='none').mean()

        # reward_dist = Normal(esti_reward, 1)
        # reward_loss = -reward_dist.log_prob(reward_batch).mean()

        self.optimizer.zero_grad()
        reward_loss.backward()
        self.optimizer.step()

        self.cnt += 1
        if (self.cnt % 10) == 0:
            # print("reward_loss:", reward_loss)
            pass


        return reward_loss.item()

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists('saving_models/'):
            os.makedirs('saving_models/')

        print('Saving models to {}'.format(path))

        torch.save({
            'reward_model_state_dict': self.reward_model.state_dict(),
            'reward_optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    # Load model parameters
    def load_model(self, path, evaluate=False):
        print('Loading models from {}'.format(path))

        checkpoint = torch.load(path)
        
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])

        if evaluate == True:
            self.reward_model.eval()
        else:
            self.reward_model.train()

import BRITS_project.models as brits_models
from BRITS_project.utils import to_var, linesubplot
import os
import numpy as np

class SAC_britsModel(object):
    def __init__(self, args, traj_seq_len, spcf_obs_size, spcf_action_size):
        physics_info = (args.num_plant, spcf_obs_size, spcf_action_size)
        self.brits_model = getattr(brits_models, args.brits_model).Model(seq_len=traj_seq_len, physics_info=physics_info).to(args.device)
        self.brits_optimizer = Adam(self.brits_model.parameters(), lr = 1e-3)
        self.device = args.device
        self.batch_size = 64
        print("args.device:", args.device)

    def estimate_missedObs(self, memory, schedule, spcf_obs_size):
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
        traj_data = to_var(traj_data, self.device)   # to tensor # to_var 함수는 현재, 무조건 cuda:0로 연결됨
        # print("traj_data:", traj_data['forward']['values'].shape)
        # print("traj_data['forward']['deltas']:", traj_data['forward']['deltas'])
        ret = self.brits_model.run_on_batch(traj_data, optimizer=None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        run_loss = ret['loss'].item()


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

        ## get missed observations
        missedObs = imputation[:, missedObs_idx, syst_obs_idx:syst_obs_idx+spcf_obs_size[schedule]]
        ground = eval_[:, missedObs_idx, syst_obs_idx:syst_obs_idx+spcf_obs_size[schedule]]

        error = np.abs(missedObs - ground).sum()
        missedObs = torch.FloatTensor(missedObs)
        self.plots(eval_, imputation, 0, batch_idx=0, title='eval_traj_1to5')

        return run_loss, missedObs, error, ground

    def train(self, memory):
        run_loss = 0.
        for idx in range(30):
            batch_data = memory.sample_batch(batch_size=self.batch_size)
            batch_data = to_var(batch_data, self.device)   # to tensor
            ret = self.brits_model.run_on_batch(batch_data, optimizer=self.brits_optimizer)

            run_loss += ret['loss'].item()
        ## for debug
        syst_obs_idx = 0
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        self.plots(eval_, imputation, syst_obs_idx, batch_idx=0, title='eval_traj_1to5_train')

        return run_loss
    
    def save_model(self, path):
        if not os.path.exists('saving_models/'):
            os.makedirs('saving_models/')

        print('Saving models to {}'.format(path))

        torch.save({
            'brits_model_state_dict': self.brits_model.state_dict(),
            'brits_optimizer_state_dict': self.brits_optimizer.state_dict()
        }, path)

    def load_model(self, path, evaluate=False):
        print('Loading models from {}'.format(path))

        checkpoint = torch.load(path)
        
        self.brits_model.load_state_dict(checkpoint['brits_model_state_dict'])
        self.brits_optimizer.load_state_dict(checkpoint['brits_optimizer_state_dict'])

        if evaluate == True:
            self.brits_model.eval()
        else:
            self.brits_model.train()

    def plots(self, eval_, imputation, syst_obs_idx, batch_idx, title, auto_open=False):
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
        results_dir = os.path.join('results', 'cartpole-swingup_default')
        linesubplot(xs=list(range(len(eval_1))), 
                    ys_list1 = [eval_1, eval_2, eval_3, eval_4, eval_5], 
                    ys_list2 = [impu_1, impu_2, impu_3, impu_4, impu_5],
                    legends1 = ['eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5'],
                    legends2 = ['esti_1', 'esti_2', 'esti_3', 'esti_4', 'esti_5'], 
                    title = title, 
                    subtitles = '',
                    rows = 5,
                    path=results_dir, 
                    xaxis='step', auto_open=auto_open)
    
    def plots2(self, eval_, imputation1, imputation2, syst_obs_idx, batch_idx, title, auto_open=False):
        eval_1 = eval_[batch_idx, :, syst_obs_idx]
        eval_2 = eval_[batch_idx, :, syst_obs_idx+1]
        eval_3 = eval_[batch_idx, :, syst_obs_idx+2]
        eval_4 = eval_[batch_idx, :, syst_obs_idx+3]
        eval_5 = eval_[batch_idx, :, syst_obs_idx+4]
        impu1_1 = imputation1[batch_idx, :, syst_obs_idx]
        impu1_2 = imputation1[batch_idx, :, syst_obs_idx+1]
        impu1_3 = imputation1[batch_idx, :, syst_obs_idx+2]
        impu1_4 = imputation1[batch_idx, :, syst_obs_idx+3]
        impu1_5 = imputation1[batch_idx, :, syst_obs_idx+4]
        impu2_1 = imputation2[batch_idx, :, syst_obs_idx]
        impu2_2 = imputation2[batch_idx, :, syst_obs_idx+1]
        impu2_3 = imputation2[batch_idx, :, syst_obs_idx+2]
        impu2_4 = imputation2[batch_idx, :, syst_obs_idx+3]
        impu2_5 = imputation2[batch_idx, :, syst_obs_idx+4]
        results_dir = os.path.join('results', 'cartpole-swingup_default')
        linesubplot(xs=list(range(len(eval_1))), 
                    ys_list1 = [eval_1, eval_2, eval_3, eval_4, eval_5], 
                    ys_list2 = [impu1_1, impu1_2, impu1_3, impu1_4, impu1_5],
                    ys_list3 = [impu2_1, impu2_2, impu2_3, impu2_4, impu2_5],
                    legends1 = ['eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5'],
                    legends2 = ['esti1_1', 'esti1_2', 'esti1_3', 'esti1_4', 'esti1_5'], 
                    legends3 = ['esti2_1', 'esti2_2', 'esti2_3', 'esti2_4', 'esti2_5'], 
                    title = title, 
                    subtitles = '',
                    rows = 5,
                    path=results_dir, 
                    xaxis='step', auto_open=auto_open)