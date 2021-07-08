###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import time
import numpy as np

import torch
import torch.nn as nn

import latent_ode_master.lib.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint

#####################################################################################################

class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
			Output:
				pred_y: tensor, [first_point.size()[0], [1], time_steps_to_predict size, [2]]
		"""
		# print("first_point size:", first_point.size())
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		assert(not torch.isnan(first_point).any()), "[diffeq_solver.py] first_point:{}  , time_steps_to_predict:{}".format(first_point, time_steps_to_predict)
		# print("first_point:", first_point)
		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# print(torch.isnan(pred_y).any())
		assert(not torch.isnan(pred_y).any()), "[diffeq_solver.py] pred_y:{},   first_point:{},   time_steps_to_predict:{}".format(pred_y, first_point, time_steps_to_predict)

		# print("pred_y shape 1:", pred_y.shape)	# tensor [time_steps_to_predict size, first_point.size()[0], [1], [2]]
		pred_y = pred_y.permute(1,2,0,3)	
		# print("pred_y shape 2:", pred_y.shape)	# tensor [first_point.size()[0], [1], time_steps_to_predict size, [2]]

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y


