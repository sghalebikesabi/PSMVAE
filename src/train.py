import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm

from utils import rmse_loss, log_normal, normal_KLD, impute, init_weights, renormalization, rounding
from models import vae, gmvae, psmvae_a, psmvae_b


model_map = {
    'VAE': vae.VAE,
    'GMVAE': gmvae.GMVAE,
    'DLGMVAE': psmvae_b.Model,
    'PSMVAEwoM': psmvae_b.Model,
    'PSMVAE_b': psmvae_b.Model,
    'PSMVAE_a': psmvae_a.Model,
}


def loss(recon, variational_params, latent_samples, data, compl_data, M_obs, M_miss, args, mode, L=1):

    data_weight = torch.sum(M_obs + M_miss)/torch.sum(M_obs).float() # adjust mse to missingness rate
    
    # mse_data = ((recon['xobs'] * M_obs.repeat((L,1)) - data.repeat((L,1)) * M_obs.repeat((L,1)))**2).sum(-1) 
    mse_data = - log_normal(data*M_obs, recon['xobs']*M_obs, torch.tensor([0.25]).to(args.device)).sum(-1)
    kld_z = normal_KLD(latent_samples['z'], variational_params['z_mu'].repeat((1,L,1)), variational_params['z_logvar'].repeat((1,L,1)), variational_params['z_mu_prior'].repeat((1,L,1)), variational_params['z_logvar_prior'].repeat((1,L,1))).sum(-1)

    recon_data_xobs = recon['xobs']
    
    if recon['M_sim_miss'] is not None:
        pi = recon['M_sim_miss']
        mpi = 1 - recon['M_sim_miss']
        mse_mask = -torch.log(pi*M_miss.float() + mpi*(1-M_miss.float())+1e-6).sum(-1)
    else: 
        mse_mask = torch.tensor(0).to(data.device).float()
 
    if variational_params['xmis_mu'] is not None:

        # kld_xmis = normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], variational_params['xmis_mu_prior'], variational_params['xmis_logvar_prior']).sum(-1)
        # kld_xmis = (variational_params['qy'].T * kld_xmis).sum(0).mean()
        
        if 'PSMVAE' not in args.model_class: 
            kld_xmis = normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device))).sum(-1)
            mse_xmis = torch.tensor([0]).to(data.device)

        elif 'PSMVAE' in args.model_class:

            # kld_xmis = M_miss*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], variational_params['xmis_mu_prior'], variational_params['xmis_logvar_prior']).sum(-1)
            # kld_xmis = M_obs*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], variational_params['xmis_mu_prior'], variational_params['xmis_logvar_prior']).sum(-1)
            kld_xmis = (M_miss*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1)
            kld_xmis += (M_obs*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1) * args.pi
            
            mse_xmis = -log_normal(data*M_obs, recon['xmis']*M_obs, torch.tensor([0.25]).to(args.device)) * (1 - args.pi)
            # mse_xmis -= log_normal(latent_samples['xmis']*M_miss, recon['xmis']*M_miss, 0.25) 
            # mse_xmis -= log_normal(latent_samples['xmis']*M_obs, recon['xmis']*M_obs, 0.25) * args.pi

            # log_xmis_prior = M_miss*log_normal(latent_samples['xmis'], variational_params['xmis_mu_prior'], torch.exp(variational_params['xmis_logvar_prior']))
            # log_xmis_prior += M_obs*log_normal(latent_samples['xmis'], variational_params['xmis_mu_prior'], torch.exp(variational_params['xmis_logvar_prior']))
                            
            # weight with pattern set probs
            mse_xmis = (variational_params['qy'].repeat((L, 1)).T * mse_xmis.sum(-1)).sum(0).mean()
            # log_xmis_prior = (variational_params['qy'].repeat((L, 1)).T * log_xmis_prior.sum(-1)).sum(0).mean()                        
        kld_xmis = (variational_params['qy'].repeat((L, 1)).T * kld_xmis).sum(0).mean()
        recon_data_xmis = torch.einsum('ir, rij -> ij', [variational_params['qy'], recon['xmis']])
    else:
        kld_xmis = torch.tensor([0]).to(data.device)
        mse_xmis = torch.tensor([0]).to(data.device)
        log_xmis_prior = torch.tensor([0]).to(data.device)
        recon_data_xmis = compl_data

    if variational_params['qy'] is not None:
        nent_r = variational_params['qy'] * torch.nn.LogSoftmax(1)(variational_params['qy_logit'])
        kld_r = (nent_r.sum(-1) - np.log(1/variational_params['qy'].shape[1])).mean()
        
        kld_z = (variational_params['qy'].repeat((L,1)).T * kld_z).sum(0).mean()
        mse_data = (variational_params['qy'].repeat((L,1)).T * mse_data).sum(0).mean()
        mse_mask = (variational_params['qy'].repeat((L,1)).T * mse_mask).sum(0).mean()

        recon_data_xobs = torch.einsum('ir, rij -> ij', [variational_params['qy'], recon_data_xobs])
    else:
        kld_r = torch.tensor(0).to(data.device).float()
        mse_data = mse_data.mean()
        mse_mask = mse_mask.mean()
        kld_z = kld_z.mean()

    loss = data_weight * mse_data + mse_mask + mse_xmis + args.z_beta * kld_z + args.r_beta * kld_r + args.xmis_beta * kld_xmis # + log_xmis_prior

    loss_dict = {                 
        mode + ' opt loss': loss,
        mode + ' z KLD':  kld_z,
        mode + ' r KLD':  kld_r,
        mode + ' miss mask MSE': mse_mask,
        mode + ' xobs MSE': mse_data,
        mode + ' xobs Imputation RMSE': torch.tensor(rmse_loss(recon_data_xobs.detach().cpu().numpy(), compl_data.cpu().numpy(), M_miss.cpu().numpy(), True)).to(data.device),
        mode + ' xmis Imputation RMSE': torch.tensor(rmse_loss(recon_data_xmis.detach().cpu().numpy(), compl_data.cpu().numpy(), M_miss.cpu().numpy(), True)).to(data.device),
    }

    return loss_dict


def train_VAE(data_train_full, data_test_full, compl_data_train_full, compl_data_test_full, wandb, args, norm_parameters):
        
    args.device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = DataLoader(TensorDataset(torch.from_numpy(data_train_full), torch.from_numpy(compl_data_train_full)), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(data_test_full), torch.from_numpy(compl_data_test_full)), batch_size=args.batch_size, shuffle=True, **kwargs)

    model = model_map[args.model_class](compl_data_train_full.shape[1], args).to(args.device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.max_epochs)):

        model.train()
        for batch_idx, (data, compl_data) in enumerate(train_loader):
            # data
            data = data.float().to(args.device)
            compl_data = compl_data.float().to(args.device)
            M_obs = (~data.isnan() & ~compl_data.isnan()).to(args.device).float()
            M_miss = (data.isnan() & ~compl_data.isnan()).to(args.device).float()
            data[data!=data] = 0
            # optimization
            optimizer.zero_grad()
            recon, variational_params, latent_samples = model(data, M_miss)
            loss_dict = loss(recon, variational_params, latent_samples, data, compl_data, M_obs, M_miss, args, 'train')
            loss_dict['train opt loss'].backward()
            optimizer.step()

        if epoch % args.log_interval == 0:
            wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()})
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, compl_data) in enumerate(test_loader):
                    # data 
                    data = data.float().to(args.device)
                    compl_data = compl_data.float().to(args.device)
                    M_obs = ~data.isnan() & ~compl_data.isnan()
                    M_miss = data.isnan() & ~compl_data.isnan()
                    data[data!=data] = 0
                    # optimization
                    recon, variational_params, latent_samples = model(data, M_miss)
                    loss_dict = loss(recon, variational_params, latent_samples, data, compl_data, M_obs, M_miss, args, 'test')
                    wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()})

    model.eval()
    
    with torch.no_grad():
        compl_data_train_full_renorm = renormalization(compl_data_train_full.copy(), norm_parameters)

        if args.model_class == 'PSMVAE_a' or args.model_class == 'PSMVAE_b':
            imp_name = 'xmis'
        else:
            imp_name = 'xobs'

        # single importance sample
        M_sim_miss_train_full = np.isnan(data_train_full) & ~np.isnan(compl_data_train_full)
        data_train_filled_full = torch.from_numpy(np.nan_to_num(data_train_full.copy(), 0)).to(args.device).float()
        recon_train, variational_params_train, latent_samples_train = model(data_train_filled_full, torch.tensor(M_sim_miss_train_full).to(args.device))
        data_test_filled_full = torch.from_numpy(np.nan_to_num(data_test_full.copy(), 0)).to(args.device).float()
        M_sim_miss_test_full = np.isnan(data_test_full) & ~np.isnan(compl_data_test_full)
        recon_test, variational_params_test, latent_samples_test = model(data_test_filled_full, torch.tensor(M_sim_miss_test_full).to(args.device))
        if variational_params_train['qy'] == None:
            recon_train['xobs'] = recon_train['xobs'].repeat((1,1,1))
            variational_params_train['qy'] = torch.ones((data_train_full.shape[0], 1))
            recon_test['xobs'] = recon_test['xobs'].repeat((1,1,1))
            variational_params_test['qy'] = torch.ones((data_test_full.shape[0], 1))
        train_imputed_1_xobs = torch.einsum("ik,kij->ij", [variational_params_train['qy'], recon_train['xobs']]) 
        test_imputed = torch.einsum("ik,kij->ij", [variational_params_test['qy'], recon_test[imp_name]])
        if recon_train['xmis'] != None:
            train_imputed_1_xmis = torch.einsum("ik,kij->ij", [variational_params_train['qy'], recon_train['xmis']]) 
        else:
            train_imputed_1_xmis = torch.from_numpy(data_train_full)
        
        if args.num_samples == 1:
            train_imputed_xobs = train_imputed_1_xobs
            train_imputed_xmis = train_imputed_1_xmis
        else:
            # multiple importance samples
            train_imputed_xobs, train_imputed_xmis = impute(model, data_train_full, args, kwargs) 

        train_imputed_xobs_ = renormalization(train_imputed_xobs.cpu(), norm_parameters)
        train_imputed_xobs_ = rounding(train_imputed_xobs_, compl_data_train_full_renorm)
        train_xobs_mis_mse = rmse_loss(train_imputed_xobs_.cpu().numpy(), compl_data_train_full_renorm, M_sim_miss_train_full)
        
        if recon_train['xmis'] != None:
            train_imputed_xmis_ = renormalization(train_imputed_xmis.cpu(), norm_parameters)
            train_imputed_xmis_ = rounding(train_imputed_xmis_, compl_data_train_full_renorm)
            train_xmis_mis_mse = rmse_loss(train_imputed_xmis_.cpu().numpy(), compl_data_train_full_renorm, M_sim_miss_train_full)
        else:
            train_xmis_mis_mse = None

        wandb.log({'xobs imp rmse': train_xobs_mis_mse, 'xmis imp rmse': train_xmis_mis_mse,})

    train_imputed, train_imputed_1, test_imputed = {
        'xobs': (train_imputed_xobs.cpu().numpy(), train_imputed_1_xobs.cpu().numpy(), test_imputed.cpu().numpy()),
        'xmis': (train_imputed_xmis.cpu().numpy(), train_imputed_1_xmis.cpu().numpy(), test_imputed.cpu().numpy())
    }[imp_name]

    return(train_imputed, train_imputed_1, test_imputed)
