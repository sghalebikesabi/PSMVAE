import argparse
import os
import numpy as np
import pandas as pd
import sys
import torch
import torch.utils.data
import wandb

from utils import make_deterministic, save_image_reconstructions
from utils import rounding, renormalization, normalization, rmse_loss

from train import train_VAE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge



model_map = {
    'miwae': None,
    'notmiwae': None,
    'missForest': IterativeImputer,
    'mice': IterativeImputer,
    'gain': None,
    'mean': 'mean',
    'VAE': train_VAE,
    'GMVAE': train_VAE,
    'DLGMVAE': train_VAE,
    'PSMVAEwoM': train_VAE,
    'PSMVAE_a': train_VAE,
    'PSMVAE_b': train_VAE,
}


def parse_args(argv):
    """Parses the arguments for the model."""
    parser = argparse.ArgumentParser(description='VAEs for missingness example')

    # input data
    parser.add_argument('--compl-data-file', nargs='?', default='data/MNIST/data_0', help='complete data file (without header, without file name ending)')
    parser.add_argument('--miss-data-file', nargs='?', default='data/MNIST/miss_data/label_uniform_frac_20_seed_0', help='missing data file (without header, without file name ending)')
    parser.add_argument('--targets-file', nargs='?', default=None, help='targets file (without header, without file name ending)')
    
    # logging results
    parser.add_argument('--results-dir', nargs='?', default='logs', help='logs directory')
    parser.add_argument('--wandb-run-name', nargs='?', default='test', help='Name for wandb Run')
    parser.add_argument('--wandb-tag', nargs='?', default='test', help='Tag for wandb Run')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status')
    parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 1)')

    # model parameters
    parser.add_argument('--model-class', nargs='?', default='PSMVAE_b', choices=model_map.keys(), help='model class, choices: ' 
                            + ' '.join(model_map.keys()))
    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 200)')
    parser.add_argument('--max-epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 1,000)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='N', help='learning rate of Adam optimizer')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--miss-mask-training', action='store_true', default=False,
                            help='incorporation of missingness mask in training')
    parser.add_argument('--num-samples', type=int, default=1, help='number of draws')
    parser.add_argument('--num-samples-train', type=int, default=1, help='number of draws in training')
    parser.add_argument('--z-dim', type=int, default=20, help='dimension of latent factor z (default: 20)')
    parser.add_argument('--r-cat-dim', type=int, default=10, help='dimension of latent factor s (default: 10)')  
    parser.add_argument('--h-dim', type=int, default=400, help='dimension of hidden layers (default: 128)')
    parser.add_argument('--pi', type=float, default=0.005, help='1-pi is probability of supervision of xmis')
    parser.add_argument('--z-beta', type=float, default=1, help='weight of z KLD')
    parser.add_argument('--r-beta', type=float, default=1, help='weight of r KLD')
    parser.add_argument('--xmis-beta', type=float, default=1, help='weight of xmis KLD')
    parser.add_argument('--iptw', action='store_true', default=False, help='IPTW weighting')
    parser.add_argument('--post-sample', action='store_true', default=False, help='dample from MICE')
    parser.add_argument('--hint-rate', help='hint probability of GAIN', default=0.9, type=float)    
    parser.add_argument('--alpha', help='hyperparameter of GAIN', default=100, type=float)

    args = parser.parse_args()
    
    args.mnist = ('MNIST' in args.compl_data_file)

    args.cuda = torch.cuda.is_available()

    if args.model_class == 'gain':
        from models import gain
        model_map['gain'] = gain.gain

    if 'miwae' in args.model_class:
        from models import notmiwae
        model_map['miwae'] = notmiwae.model
        model_map['notmiwae'] = notmiwae.model
        args.z_dim = 1
        args.num_samples_train = 50

    if 'IPTW' in args.model_class:
        args.iptw = True

    if (('PSMVAE_a' == args.model_class) or ('PSMVAE_b' == args.model_class)):
        args.miss_mask_training = True
    elif ('PSMVAEwoM' == args.model_class):
        args.miss_mask_training = False

    if args.model_class == 'mice': #TODO
        args.num_samples = 1

    if args.cuda:
        args.log_interval = 1000

    return(args)


def main(args): 

    # load data 
    if args.targets_file is not None:
        files = [args.compl_data_file, args.miss_data_file, args.targets_file]
    else:
        files = [args.compl_data_file, args.miss_data_file]
    train_files, val_files, test_files = [[file + ending for file in files] for ending in ['.train', '.val', '.test']]

    compl_data_train_ori = pd.read_csv(train_files[0], header=None, dtype=float)
    compl_data_val_ori = pd.read_csv(val_files[0], header=None, dtype=float)
    compl_data_test_ori = pd.read_csv(test_files[0], header=None, dtype=float)
    M_sim_miss_train = pd.DataFrame(pd.read_csv(train_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_train_ori), dtype=bool)
    M_sim_miss_val = pd.DataFrame(pd.read_csv(val_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_val_ori), dtype=bool)
    M_sim_miss_test = pd.DataFrame(pd.read_csv(test_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_test_ori), dtype=bool)
    data_train_ori = compl_data_train_ori.mask(M_sim_miss_train)
    data_val_ori = compl_data_val_ori.mask(M_sim_miss_val)
    data_test_ori = compl_data_test_ori.mask(M_sim_miss_test)
    try:
        targets_train = pd.read_csv(train_files[2], header=None, dtype=float)
        targets_val = pd.read_csv(val_files[2], header=None, dtype=float)
        targets_test = pd.read_csv(test_files[2], header=None, dtype=float)
    except IndexError:
        targets_train, targets_val, targets_test = None, None, None
    M_sim_miss_train = M_sim_miss_train.values
    M_sim_miss_val = M_sim_miss_val.values
    M_sim_miss_test = M_sim_miss_test.values
    
    # normalize data 
    norm_type = 'minmax' * args.mnist + 'standard' * (1-args.mnist)
    data_train, norm_parameters = normalization(data_train_ori.values,None, norm_type)
    data_val, _ = normalization(data_val_ori.values, norm_parameters, norm_type)
    data_test, _ = normalization(data_test_ori.values, norm_parameters, norm_type)

    compl_data_train, _ = normalization(compl_data_train_ori.values, None, norm_type)
    compl_data_val, _ = normalization(compl_data_val_ori.values, norm_parameters, norm_type)
    compl_data_test, _ = normalization(compl_data_test_ori.values, norm_parameters, norm_type)

    # logging
    make_deterministic(args.seed)
    wandb.init(project="miss-vae", name=args.wandb_run_name, tags=[args.wandb_tag], save_code=True)
    wandb.config.update(args)

    # compute imputations
    if (args.model_class=='mice') or (args.model_class=='missForest'):
        train_imputed = np.zeros_like(data_train)
        test_imputed = np.zeros_like(data_test)
        for l in range(args.num_samples):
            if args.model_class == 'mice':
                imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=BayesianRidge(), sample_posterior=(args.no_post_sample==False)) 
            elif args.model_class == 'missForest':
                imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=ExtraTreesRegressor(n_estimators=10, n_jobs=2)) 
            train_imputed += imputer.fit_transform(data_train)
            test_imputed += imputer.transform(data_test)
        train_imputed /= args.num_samples
        test_imputed /= args.num_samples
    else:
        if args.model_class == 'mean':
            data_train_df = pd.DataFrame(data_train)
            data_test_df = pd.DataFrame(data_test)
            train_imputed = data_train_df.fillna(data_train_df.mean(), inplace=False).values
            test_imputed = data_test_df.fillna(data_train_df.mean(), inplace=False).values
        elif args.model_class == 'gain':
            gain_config = {
                'batch_size': args.batch_size,
                'hint_rate': args.hint_rate,
                'alpha': args.alpha,
                'iterations': args.max_epochs,
                'num_samples': args.num_samples,
            }
            train_imputed, test_imputed = model_map[args.model_class](data_train, data_test, gain_config, args)
            train_imputed = np.mean(train_imputed, axis=0)            
            test_imputed = np.mean(test_imputed, axis=0)            
        if 'VAE' in args.model_class:
            train_imputed, train_imputed_1, test_imputed = model_map[args.model_class](data_train, data_test, compl_data_train, compl_data_test, wandb, args, norm_parameters)
        elif args.model_class == 'miwae':
            train_imputed, test_imputed = model_map[args.model_class](compl_data_train, data_train, compl_data_test, compl_data_test, norm_parameters, wandb, args)

    if args.mnist:
        save_image_reconstructions(train_imputed*M_sim_miss_train+compl_data_train*(1-M_sim_miss_train), compl_data_train, M_sim_miss_train, 28, 'images', args.model_class + "_" + args.miss_data_file.split('/')[-1].split('_')[0])
    
    # compute losses
    M_obs_train, M_obs_test = ~M_sim_miss_train & ~np.isnan(compl_data_train), ~M_sim_miss_test & ~np.isnan(compl_data_test)

    # renormalization
    train_imputed = renormalization(train_imputed, norm_parameters, norm_type)
    test_imputed = renormalization(test_imputed, norm_parameters, norm_type)
    compl_data_train = renormalization(compl_data_train, norm_parameters, norm_type)
    compl_data_test = renormalization(compl_data_test, norm_parameters, norm_type)

    # rounding
    train_imputed = rounding(train_imputed, compl_data_train)
    test_imputed = rounding(test_imputed, compl_data_test)

    # save imputations
    try:
        imputed_dir = '/'.join(args.compl_data_file.split('/')[:-1] + ['imputed'])
        from pathlib import Path
        Path(imputed_dir).mkdir(parents=True, exist_ok=True)    
    except FileExistsError:
        pass
    np.savetxt(imputed_dir + f'/{args.wandb_run_name}.train', train_imputed, delimiter=',')
    np.savetxt(imputed_dir + f'/{args.model_class}.test', test_imputed, delimiter=',')
    
    # compute loss
    train_mis_mse = rmse_loss(train_imputed, compl_data_train, M_sim_miss_train)
    test_mis_mse = rmse_loss(test_imputed, compl_data_test, M_sim_miss_test)

    # log loss
    wandb.log({'Train Imputation RMSE loss': train_mis_mse})
    wandb.log({'Test Imputation RMSE loss': test_mis_mse})

    # loss for a single importance sample
    if 'VAE' in args.model_class:
        train_imputed_1 = renormalization(train_imputed_1, norm_parameters, norm_type)
        train_imputed_1 = rounding(train_imputed_1, compl_data_train)
        train_1_mis_mse = rmse_loss(train_imputed_1, compl_data_train, M_sim_miss_train)
        wandb.log({'Train Imputation RMSE loss (single sample)': train_1_mis_mse})

    with open('table.txt', "a") as myfile:
        myfile.write(','.join(map(str, [args.miss_data_file, args.seed, args.model_class, train_mis_mse, test_mis_mse])) + '\n')

    print(','.join(map(str, [args.miss_data_file, args.seed, args.model_class, train_mis_mse, test_mis_mse])))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:]) # sys.argv[0] is file name
    main(args)
