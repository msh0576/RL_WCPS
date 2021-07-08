import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import BRITS_project.utils as utils
import BRITS_project.models as models
import argparse
import BRITS_project.data_loader as data_loader
import pandas as pd
import ujson as json
import pickle

from sklearn import metrics

from ipdb import set_trace

from BRITS_project.utils import lineplot, linesplot
import os

def hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--model', type = str)
    args = parser.parse_args()
    return args

def train(args, model, path, log_path, model_):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_iter = data_loader.get_loader(batch_size = args.batch_size)
    
    run_losses = []
    MAEs = []
    for epoch in range(args.epochs):
        # print("model feat_reg state_dict:", model.feat_reg.state_dict())
        print('epoch:', epoch)
        model.train()

        run_loss = 0.0
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)

            ret = model.run_on_batch(data, optimizer)
            # print("ret:", ret)

            # run_loss += ret['loss'].data[0]
            run_loss += ret['loss'].item()
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))
        run_losses.append(run_loss)

        if epoch % 1 == 0:
            MAE, info = evaluate(model, data_iter)
            MAEs.append(MAE)
            if epoch == args.epochs-1:  # trajectory info store
                info['model'] = args.model
                with open(log_path, 'wb') as pkl_file:
                    pickle.dump(info, pkl_file)
    lineplot(list(range(len(run_losses))), run_losses, title='{}_Loss'.format(model_), path=path, xaxis='epoch')
    lineplot(list(range(len(MAEs))), MAEs, title='{}_MAE'.format(model_), path=path, xaxis='epoch')

def utilize(model, traj):
    '''
        Input:
            model: [1, time_sequnece, obs_size]
            traj: trajectory values
        Output:
            obs: a missed observation
    '''
    ret = model.run_on_batch(traj)

    return ret

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # print("current time:", ret['times'])
        # print("current evals:", ret['evals'][:,0,:])

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        # pred = pred[np.where(is_train == 0)]
        # label = label[np.where(is_train == 0)]

        # labels += label.tolist()
        # preds += pred.tolist()

    # labels = np.asarray(labels).astype('int32')
    # preds = np.asarray(preds)

    # print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    # Graph of imputations
    # print("imputations:", len(imputations))
    # print("eval_:", eval_)

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    MAE = np.abs(evals - imputations).mean()
    MRE = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    print('MAE', MAE)
    print('MRE', MRE)

    info = {
        'imputation': imputation,
        'eval': eval_
    }

    return MAE, info

def figure_log_traj(path, *models):

    models_log = []
    for model in models:
        file_name = '/{}_log'.format(model)
        log_path = path + file_name

        # whether exist input model's log
        if os.path.exists(log_path):
            pass
        else:
            print("Doesn't exist {}".format(file_name))
            return

        # extract log
        with open(log_path, 'rb') as pkl_file:
            models_log.append(pickle.load(pkl_file))
    
    for model_log in models_log:
        file_name = '/{}_log'.format(model)
        log_path = path + file_name

        print("model name:", model_log['model'])
        print("model_log['imputation'] shape:", model_log['imputation'].shape)

        batch_idx = 10
        system_idx = 0
        each_obs_size = 5
        obs_idx = 0
        obs_traj = model_log['imputation'][batch_idx, :, each_obs_size*system_idx + obs_idx]
        eval_traj = model_log['eval'][batch_idx, :, each_obs_size*system_idx + obs_idx]
        linesplot(list(range(len(obs_traj))), [obs_traj, eval_traj], legends=['obs_traj', 'eval_traj'], title='{}_traj'.format(model_log['model']), path=path, xaxis='step')


    

def run(args):
    model = getattr(models, args.model).Model()
    results_dir = os.path.join("BRITS_project","results", "SAC_estimate_missedObservations")
    os.makedirs(results_dir, exist_ok=True)

    if torch.cuda.is_available():
        model = model.cuda()
    
    log_path = results_dir + '/{}_log'.format(args.model)
    # train(args, model, results_dir, log_path, args.model)
    figure_log_traj(results_dir, 'rits_i', 'brits_i', 'rits', 'brits')


if __name__ == '__main__':
    args = hyperparameters()
    run(args)
    
