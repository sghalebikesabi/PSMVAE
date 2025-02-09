import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    'hivae': None,
    'mean': 'mean',
    'VAE': train_VAE,
    'GMVAE': train_VAE,
    'DLGMVAE': train_VAE,
    'PSMVAEwoM': train_VAE,
    'PSMVAE_a': train_VAE,
    'PSMVAE_b': train_VAE,
    'PSMVAE_c': train_VAE,
}


def parse_args(argv):
    """Parses the arguments for the model."""
    parser = argparse.ArgumentParser(description='VAEs for missingness example')

    # input data
    parser.add_argument('--compl-data-file', nargs='?', default='data/adult/data_0', help='complete data file (without header, without file name ending)')
    parser.add_argument('--miss-data-file', nargs='?', default='data/adult/miss_data/MCAR_notuniform_frac_20_seed_0', help='missing data file (without header, without file name ending)')
    parser.add_argument('--targets-file', nargs='?', default=None, help='targets file (without header, without file name ending)')
    
    # logging results
    parser.add_argument('--results-dir', nargs='?', default='logs', help='logs directory')
    parser.add_argument('--wandb-run-name', nargs='?', default='test', help='Name for wandb Run')
    parser.add_argument('--wandb-tag', nargs='?', default='test', help='Tag for wandb Run')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status')
    parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 1)')

    # model parameters
    parser.add_argument('--model-class', nargs='?', default='hivae', choices=model_map.keys(), help='model class, choices: ' 
                            + ' '.join(model_map.keys()))
    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 200)')
    parser.add_argument('--max-epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 1,000)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='N', help='learning rate of Adam optimizer')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--miss-mask-training', action='store_true', default=False,
                            help='incorporation of missingness mask in training')  
    parser.add_argument('--num-samples', type=int, default=10, help='number of draws')
    parser.add_argument('--num-samples-train', type=int, default=1, help='number of draws in training')
    parser.add_argument('--z-dim', type=int, default=20, help='dimension of latent factor z (default: 20)')
    parser.add_argument('--r-cat-dim', type=int, default=10, help='dimension of latent factor s (default: 10)')  
    parser.add_argument('--h-dim', type=int, default=400, help='dimension of hidden layers (default: 128)')
    parser.add_argument('--pi', type=float, default=0.005, help='1-pi is probability of supervision of xmis')
    parser.add_argument('--z-beta', type=float, default=1, help='weight of z KLD')
    parser.add_argument('--r-beta', type=float, default=1, help='weight of r KLD')
    parser.add_argument('--xmis-beta', type=float, default=1, help='weight of xmis KLD')
    parser.add_argument('--downstream-logreg', action='store_true', default=False, help='should downstream logistic regression be run on targets')
    parser.add_argument('--iptw', action='store_true', default=False, help='IPTW weighting')
    parser.add_argument('--post-sample', action='store_true', default=False, help='dample from MICE')
    parser.add_argument('--hint-rate', help='hint probability of GAIN', default=0.9, type=float)    
    parser.add_argument('--alpha', help='hyperparameter of GAIN', default=100, type=float)
    parser.add_argument('--mul-imp', action='store_true', default=False, help='multiple imputation') 
    parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')

    args = parser.parse_args()
    
    args.mnist = ('MNIST' in args.compl_data_file)

    args.cuda = torch.cuda.is_available()
    print(args.cuda)

    if args.model_class == 'gain':
        from models import gain
        model_map['gain'] = gain.gain

    elif 'miwae' in args.model_class:
        from models import miwae
        model_map['miwae'] = miwae.notMiwae
        model_map['notmiwae'] = miwae.notMiwae
        args.z_dim = 1
        args.num_samples_train = 50

    elif args.model_class == 'hivae':
        from models import hivae
        model_map['hivae'] = hivae.hivae

    if 'IPTW' in args.model_class:
        args.iptw = True

    if (('PSMVAE_a' == args.model_class) or ('PSMVAE_b' == args.model_class)):
        args.miss_mask_training = True
    elif ('PSMVAEwoM' == args.model_class):
        args.miss_mask_training = False

    if (args.model_class == 'mice') and not args.mul_imp: #TODO
        args.num_samples = 1

    if args.mul_imp:
        args.post_sample = True

    if args.cuda:
        args.log_interval = 1

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
        targets_train = np.squeeze(pd.read_csv(train_files[2], header=None).values) 
        targets_val = np.squeeze(pd.read_csv(val_files[2], header=None).values) 
        targets_test = np.squeeze(pd.read_csv(test_files[2], header=None).values) 
        target_unq = np.unique(targets_train)
        
        ohe_targets_train = np.zeros((len(targets_train), len(target_unq)))
        ohe_targets_val = np.zeros((len(targets_val), len(target_unq)))
        ohe_targets_test = np.zeros((len(targets_test), len(target_unq)))
        for targeti in range(len(target_unq)):
            ohe_targets_train[targets_train==target_unq[targeti], targeti] = 1
            ohe_targets_val[targets_val==target_unq[targeti], targeti] = 1
            ohe_targets_test[targets_test==target_unq[targeti], targeti] = 1

    except IndexError:
        targets_train, targets_val, targets_test = None, None, None
    M_sim_miss_train = M_sim_miss_train.values
    M_sim_miss_val = M_sim_miss_val.values
    M_sim_miss_test = M_sim_miss_test.values
    
    # normalize data 
    norm_type = 'minmax' * args.mnist + 'standard' * (1-args.mnist)
    data_train, norm_parameters = normalization(data_train_ori.values, None, norm_type)
    data_val, _ = normalization(data_val_ori.values, norm_parameters, norm_type)
    data_test, _ = normalization(data_test_ori.values, norm_parameters, norm_type)

    compl_data_train, _ = normalization(compl_data_train_ori.values, norm_parameters, norm_type)
    compl_data_val, _ = normalization(compl_data_val_ori.values, norm_parameters, norm_type)
    compl_data_test, _ = normalization(compl_data_test_ori.values, norm_parameters, norm_type)

    # logging
    make_deterministic(args.seed)
    wandb.init(project="miss-vae", name=args.wandb_run_name, tags=[args.wandb_tag], save_code=True)
    wandb.config.update(args)

    # compute imputations
    if (args.model_class=='mice') or (args.model_class=='missForest'):
        train_imputed = []
        test_imputed = []
        for l in range(args.num_samples):
            if args.model_class == 'mice':
                imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=BayesianRidge(), sample_posterior=args.post_sample) 
            elif args.model_class == 'missForest':
                imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=ExtraTreesRegressor(n_estimators=10, n_jobs=2)) 
            train_imputed.append(imputer.fit_transform(data_train))
            test_imputed.append(imputer.transform(data_test))
        if not args.mul_imp:
            train_imputed = np.mean(train_imputed, axis=0) 
            test_imputed = np.mean(test_imputed, axis=0)
    else:
        if args.model_class == 'hivae':
            train_imputed, test_imputed = model_map[args.model_class](data_train, data_test, M_sim_miss_train, M_sim_miss_test, args)
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
            if not args.mul_imp:
                train_imputed = np.mean(train_imputed, axis=0)            
                test_imputed = np.mean(test_imputed, axis=0)            
        if 'VAE' in args.model_class:
            train_imputed, train_imputed_1, test_imputed = model_map[args.model_class](data_train, data_test, compl_data_train, compl_data_test, wandb, args, norm_parameters)
        elif (args.model_class == 'miwae') or (args.model_class == 'notmiwae'):
            train_imputed, test_imputed = model_map[args.model_class](compl_data_train, data_train, compl_data_test, compl_data_test, norm_parameters, wandb, args)

    if args.mnist:
        save_image_reconstructions(train_imputed*M_sim_miss_train+compl_data_train*(1-M_sim_miss_train), compl_data_train, M_sim_miss_train, 28, 'images', args.wandb_run_name)
    
    # compute losses
    M_obs_train, M_obs_test = ~M_sim_miss_train & ~np.isnan(compl_data_train), ~M_sim_miss_test & ~np.isnan(compl_data_test)

    if not args.mul_imp:
        # renormalization
        train_imputed = renormalization(train_imputed, norm_parameters, norm_type)
        test_imputed = renormalization(test_imputed, norm_parameters, norm_type)
        compl_data_train = renormalization(compl_data_train, norm_parameters, norm_type)
        compl_data_test = renormalization(compl_data_test, norm_parameters, norm_type)

        # rounding
        train_imputed = rounding(train_imputed, compl_data_train)
        test_imputed = rounding(test_imputed, compl_data_test)
    else:
        # renormalization
        train_imputed = [renormalization(train_imputed[i], norm_parameters, norm_type) for i in range(len(train_imputed))]
        test_imputed = [renormalization(test_imputed[i], norm_parameters, norm_type) for i in range(len(test_imputed))]
        compl_data_train = renormalization(compl_data_train, norm_parameters, norm_type)
        compl_data_test = renormalization(compl_data_test, norm_parameters, norm_type)

        # rounding
        train_imputed = [rounding(train_imputed[i], compl_data_train) for i in range(len(train_imputed))]
        test_imputed = [rounding(test_imputed[i], compl_data_test) for i in range(len(test_imputed))]

    # save imputations
    if not args.mul_imp:
        try:
            imputed_dir = '/'.join(args.compl_data_file.split('/')[:-1] + ['imputed'])
            from pathlib import Path
            Path(imputed_dir).mkdir(parents=True, exist_ok=True)    
        except FileExistsError:
            pass
        np.savetxt(imputed_dir + f'/{args.wandb_run_name}.train', train_imputed, delimiter=',')
        np.savetxt(imputed_dir + f'/{args.model_class}.test', test_imputed, delimiter=',')
    
        # compute loss
        if args.model_class == 'hivae':
            train_mis_mse = rmse_loss(train_imputed, compl_data_train[:len(train_imputed)], M_sim_miss_train[:len(train_imputed)])
            test_mis_mse = rmse_loss(test_imputed, compl_data_test[:len(test_imputed)], M_sim_miss_test[:len(test_imputed)])
        else:
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

    else:
        data_x_train = np.concatenate(train_imputed, 0)
        data_x_test = np.concatenate(test_imputed, 0)
        if args.downstream_logreg:
            targets_train_full = np.tile(ohe_targets_train, (args.num_samples,1))
            targets_test_full = np.tile(ohe_targets_test, (args.num_samples,1))
            clf = LogisticRegression(random_state=args.seed).fit(data_x_train, targets_train_full.argmax(1))
            test_acc = clf.score(data_x_test, targets_test_full.argmax(1))

            wandb.log({'Test accuracy': test_acc})
            print(test_acc)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:]) # sys.argv[0] is file name
    main(args)
