import pdb
import torch
from tqdm import trange
from functools import partial
from collections import defaultdict


from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence

from utils import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator

def train(memory, rssm, optimizer, device, N=32, H=50, beta=1.0, grads=False):
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Varioational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device)*3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    # print("x. shape:", x.shape) # tensor, [51, 32, 3, 64, 64]
    # print("u shape:", u.shape)    # tensor, [50, 32, 1]
    preprocess_img(x, depth=5)
    e_t = bottle(rssm.encoder, x)
    # print("e_t shape:", e_t.shape)  # tensor, [51, 32, 1024]
    h_t, s_t = rssm.get_init_state(e_t[0])
    # print("s_t shape:", s_t.shape)          # tensor, [32 (batch), 30]
    # print("h_t shape:", h_t.shape)          # tensor, [32 (batch), 200]
    kl_loss, rc_loss, re_loss = 0, 0, 0
    states, priors, posteriors, posterior_samples = [], [], [], []
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
    # len(priors)=50,  priors[-1][0].shape = (32, 30) = priors[-1][1]
    # zip(*priors) : ( (mean_priors), (std_priors) ) 로 분할해서 묶임. mean_priors 묶음은 길이 50, 각 요소당 (32, 30) tensor
    # len(list(zip(*priors))[0]) : 50 (=H)
    # list(zip(*priors))[0][0].shape: tensor, [32, 30]
    # list(map(torch.stack, zip(*posteriors)))[0].shape: tensor, [50, 32, 30]: 평균 정보
    # list(map(torch.stack, zip(*posteriors)))[1].shape: tensor, [50, 32, 30]: 표준편차 정보
    ### *map(torch.stack, zip(*priors)) 목적:
    ### zip(*priors)에 의해 priors의 평균/표편 정보 50개를 묶음. 
    ### But, 각 묶음은 50개의 iterator 정보이기때문에, 이를 50 사이즈이 tensor로 합치기위해 map(torch.stack, ..)을 사용하여 [50, 32, 30]으로 만듦.
    ### 그리고, 평균 [50, 32, 30] and 표편 [50, 32, 30] 을 하나의 튜플로 묶기 위해 *map(...) 사용함.
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    # 여기서 states는 belief or hidden으로 봐도 괜찮을 듯
    states, posterior_samples = map(torch.stack, (states, posterior_samples))   # states: list len 50 -> tensor, [50, 32, 200] | posterior_samples: list -> tensor
    
    rec_loss = F.mse_loss(
        bottle(rssm.decoder, states, posterior_samples), x[1:],
        reduction='none'
    ).sum((2, 3, 4)).mean()
    # kl_divergence(posterior_dist, prior_dist).shape    # tensor, [50, 32, 30]
    # kl_divergence(posterior_dist, prior_dist).sum(-1).shape # tensor, [50, 32]
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1),
        free_nats
    ).mean()    # scalar
    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states, posterior_samples), r
    )
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)
    (beta*kld_loss + rec_loss + rew_loss).backward()
    optimizer.step()
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'reconstruction': rec_loss.item(),
            'reward_pred': rew_loss.item()
        },
    }
    if grads:
        metrics['grad_norms'] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def main():
    env = TorchImageEnvWrapper('Pendulum-v0', bit_depth=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)
    policy = RSSMPolicy(
        rssm_model, 
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=100,
        device=device
    )
    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=100,
    )
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True))
    res_dir = 'results/'
    summary = TensorBoardMetrics(f'{res_dir}/')
    for i in trange(1, desc='Epoch', leave=False):
        metrics = {}
        for _ in trange(150, desc='Iter ', leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics.keys():
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f'{k}_mean'] = np.array(v).mean()
        
        summary.update(metrics)
        mem.append(rollout_gen.rollout_once(explore=True))
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        mem.append(eval_episode)
        save_video(eval_frames, res_dir, f'vid_{i+1}')
        summary.update(eval_metrics)

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f'{res_dir}/ckpt_{i+1}.pth')

    # pdb.set_trace()

if __name__ == '__main__':
    main()
