
import latent_ode_master.lib.utils as ode_utils
import torch.nn as nn
from latent_ode_master.lib.ode_rnn import *
from latent_ode_master.lib.ode_func import ODEFunc
from latent_ode_master.lib.diffeq_solver import DiffeqSolver

def train_latentODE(args, models, optimizers, memories, batch_size):
    '''
    Input:
        data_obj: dict, {
                        'observed_data': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                        'observed_tp': tensor(GPU), [100,]
                        'data_to_predict': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                        'tp_to_predict': tensor(GPU), [100,]
                        'observed_mask': tensor(GPU), [50 (batch), 100 (traj), 14 (env dim)]
                        'mask_predicted_data': None
                        'mode': interp
                        'labels': None
                        }
    '''
    print("train_latentODE")
    # num_batches = data_obj["n_train_batches"]
    num_batches = batch_size

    train_results = {
        'loss':[],
        'mse':[],
        'observed_tp':[],
        'observed_data':[]
    }
    for itr in range(1, 5 * (args.niters + 1)):
        if itr % 100 == 0:
            print("itr:", itr)
        [optimizer.zero_grad() for optimizer in optimizers]
        [ode_utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10) for optimizer in optimizers]
        
        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        # batch_dict = ode_utils.get_next_batch(data_obj["train_dataloader"])

        tmp_loss = 0
        tmp_mse = 0
        for idx_ in range(len(models)):
            batch_dict = memories[idx_].generate_train_dataset(batch_size=num_batches)
            
            train_res = models[idx_].compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
            train_res["loss"].backward()
            optimizers[idx_].step()

            if idx_ == 0:
                # print("batch_dict['observed_tp']:", batch_dict['observed_tp'])
                tmp_loss += train_res["loss"].detach().cpu().item()
                tmp_mse += train_res['mse'].detach().cpu().item()
        train_results['loss'].append(tmp_loss)
        train_results['mse'].append(tmp_mse)
        train_results['observed_tp'] = batch_dict['observed_tp'].detach().cpu().numpy()
        train_results['observed_data'] = batch_dict['observed_data'].detach().cpu().numpy()
        train_results['pred_y'] = train_res['pred_y'][0,:,:,:].cpu().numpy()

    print("End train_latentODE!!")
    return train_results

def test_latentODE(args, models, optimizers, memories, batch_size):
    pass



def create_ODE_RNN_model(args, input_dim, obsrv_std, device, classif_per_tp=False, n_labels=1):

    n_ode_gru_dims = args.latents
                
    ode_func_net = ode_utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
        n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
    
    rec_ode_func = ODEFunc(
        input_dim = input_dim, 
        latent_dim = n_ode_gru_dims,
        ode_func_net = ode_func_net,
        device = device).to(device)

    z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
    model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
                        z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
                        concat_mask = True, obsrv_std = obsrv_std,
                        use_binary_classif = args.classif,
                        classif_per_tp = classif_per_tp,
                        n_labels = n_labels,
                        train_classif_w_reconstr = (args.dataset == "physionet")
                        ).to(device)
    return model
