import os
import numpy as np
import torch

from BRITS_project.utils import to_var, linesubplot

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


    return missedObs, error.sum(), run_loss, ground


def memorize_esti_obs(memory_list, buff_list, current_sched, initialized_buff, esti_obs, esti_reward, state, command_action, mask, num_plant, spcf_obs_size):
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
    assert current_sched >= 0 and current_sched < num_plant, 'memory_list index error'

    if initialized_buff[current_sched] == True:
        memory_list[current_sched].push(buff_list[current_sched]['obs'],
                                        buff_list[current_sched]['action'],
                                        esti_reward,
                                        esti_obs,
                                        torch.tensor([mask], dtype=torch.float32))
    ## define a new sample
    spcf_obs_crite = sum(spcf_obs_size[:current_sched]) if current_sched != 0 else 0
    # buff_list[current_sched]['obs'] = state[:, spcf_obs_crite:spcf_obs_crite+spcf_obs_size[current_sched]]
    # buff_list[current_sched]['action'] = command_action[current_sched].detach().cpu().view(1,-1)
    store_current_info(buff_list[current_sched], state[:, spcf_obs_crite:spcf_obs_crite+spcf_obs_size[current_sched]], 
                        command_action[current_sched].detach().cpu().view(1,-1))
    initialized_buff[current_sched] = True

def memorize_delayed_info(curr_memory, curr_buff, obs, command, esti_reward, esti_obs, mask):
    '''
    Input:
        obs: tensor(CPU), [1, 5(each_obs_size)]
        command: tensor(CPU), [1, 1]
        esti_reward: tensor(CPU), [1,]
        esti_obs: tensor(CPU), [1, 5]
    '''
    curr_memory.push(curr_buff['obs'],
                    curr_buff['action'],
                    esti_reward,
                    esti_obs,
                    torch.tensor([mask], dtype=torch.float32))
    store_current_info(curr_buff, obs, command)


def store_current_info(buff, obs, command):
    buff['obs'] = obs
    buff['command'] = command


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


def to_state_cash(state_cash, time_step, obs, command_action, mask):
    '''
    Input:
        state_cash: it is for preserving previous state info
        time_step: scalar
        obs: tensor [1, spcf_obs_size]
        command_action: tensor(CPU), [1, 1(spcf_action_size)]
        mask: scalar
    '''
    state_cash['time_step'] = time_step
    state_cash['obs'] = obs
    state_cash['action'] = command_action
    state_cash['mask'] = mask

def check_do_estimate(cashed_ts, curr_ts):
    '''
    it doesn't need an estimation when the observation is taken at the previous step 
    '''
    if cashed_ts == curr_ts - 1:
        do = False
    elif cashed_ts < curr_ts  - 1:
        do = True
    else:
        raise Exception('check_do_estimate() Error!')
    return do

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
                xaxis='step')
    