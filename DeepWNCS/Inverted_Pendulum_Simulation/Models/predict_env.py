
import numpy as np

class PredictEnv:
    def __init__(self, model):
        '''
            <input>
            model   : environment model
        '''
        self.model = model

    def _get_logprob(self, x, means, variances):
        '''
            <input>
            x               : [batch_size, state_dim + reward_dim]
            means/variances : [network_size, batch_size, state_dim + reward_dim]
        '''
        k = x.shape[-1] # [batch_size, 6 (state + reward)] -> 6

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        '''
            <input>
            all samples of state and action in the env_pool
            obs : [batch_size, state_dim]
            act : [batch_size, action_dim]
            -------
            ensemble_means  : [network_size (= # models), batch_size, state_dim + reward_dim],
                                where [:,:,0] = reward_mean, [:,:,1:] = delta_state_mean

        '''
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False
            
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:,:,1:] += obs   # next_state = state + delta_state
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)   # [batch_size, ]
        batch_idxes = np.arange(0, batch_size)
        # In the ensemble matrix, this is like to shuffle the matrix
        samples = ensemble_samples[model_idxes, batch_idxes]    # [batch_size, state_dim + reward_dim]
        model_means = ensemble_model_means[model_idxes, batch_idxes]    # [batch_size, state_dim + reward_dim]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.termination_fn(next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info


    def termination_fn(self, next_obs):
        POSITION_BOUNDARY = 10
        THETA_ERROR_MAX = 10.    # degree? radian?

        theta = next_obs[:, 1]
        theta_ref = 0.
        theta_err = np.abs(theta_ref - (180.0/np.pi * theta))
        not_done = np.isfinite(next_obs).all(axis=-1) \
                    * (theta_err <= THETA_ERROR_MAX)

        done = ~not_done
        done = done[:, None]    # (batch_size,) -> (batch_size,1)
        return done
