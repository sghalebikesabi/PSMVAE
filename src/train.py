import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import rmse_loss, log_normal, normal_KLD, impute, init_weights, renormalization, rounding, save_image_reconstructions_with_mask
from models import vae, gmvae, psmvae_a, psmvae_b, psmvae_c

model_map = {
    'VAE': vae.VAE,
    'GMVAE': gmvae.GMVAE,
    'DLGMVAE': psmvae_b.Model,
    'PSMVAEwoM': psmvae_b.Model,
    'PSMVAE_b': psmvae_b.Model,
    'PSMVAE_a': psmvae_a.Model,
    'PSMVAE_c': psmvae_c.Model,
}


def loss(recon, variational_params, latent_samples, data, compl_data, M_obs, M_miss, args, mode, L=1):

    data_weight = torch.sum(M_obs + M_miss)/torch.sum(M_obs).float() # adjust mse to missingness rate
    
    if args.mnist:
        if variational_params['qy'] is not None:
            mse_data = torch.stack([torch.nn.functional.binary_cross_entropy(recon['xobs'][i]*M_obs, data*M_obs, reduction='none').sum(-1) for i in range(args.r_cat_dim)]) 
        else:
            mse_data = torch.nn.functional.binary_cross_entropy(recon['xobs']*M_obs, data*M_obs, reduction='none').sum(-1)
    else:
        mse_data = - log_normal(data*M_obs, recon['xobs']*M_obs, torch.tensor([0.25]).to(args.device)).sum(-1)
    kld_z = normal_KLD(latent_samples['z'], variational_params['z_mu'].repeat((1,L,1)), variational_params['z_logvar'].repeat((1,L,1)), variational_params['z_mu_prior'].repeat((1,L,1)), variational_params['z_logvar_prior'].repeat((1,L,1))).sum(-1)
    
    if recon['M_sim_miss'] is not None:
        # pi = recon['M_sim_miss']
        # mpi = 1 - recon['M_sim_miss']
        # mse_mask = -torch.log(pi*M_miss.float() + mpi*(1-M_miss.float())+1e-6).sum(-1)
        if variational_params['qy'] is not None:
            mse_mask = torch.stack([torch.nn.functional.binary_cross_entropy(recon['M_sim_miss'][i], M_miss.float(), reduction='none').sum(-1) for i in range(args.r_cat_dim)])  
        else:
            mse_mask = torch.nn.functional.binary_cross_entropy(recon['M_sim_miss'], M_miss.float(), reduction='none').sum(-1) 
    else: 
        mse_mask = torch.tensor(0).to(data.device).float()
 
    if variational_params['xmis_mu'] is not None:
        
        if 'PSMVAE' not in args.model_class: 
            kld_xmis = normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device))).sum(-1)
            mse_xmis = torch.tensor([0]).to(data.device)

        elif 'PSMVAE' in args.model_class:
            if args.mnist:
                # kld_xmis = (M_miss*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1)
                kld_xmis = (M_miss*torch.nn.functional.binary_cross_entropy_with_logits(latent_samples['xmis'], torch.sigmoid(variational_params['xmis_mu']))).sum(-1)
                kld_xmis += (M_obs*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1) * args.pi
                
                mse_xmis = -log_normal(data*M_obs, recon['xmis']*M_obs, torch.tensor([0.25]).to(args.device)) * (1 - args.pi)
            else:
                kld_xmis = (M_miss*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1)
                kld_xmis += (M_obs*normal_KLD(latent_samples['xmis'], variational_params['xmis_mu'], variational_params['xmis_logvar'], recon['xmis'], torch.sqrt(torch.tensor([0.25]).to(args.device)))).sum(-1) * args.pi
                
                mse_xmis = -log_normal(data*M_obs, recon['xmis']*M_obs, torch.tensor([0.25]).to(args.device)) * (1 - args.pi)
                            
            # weight with pattern set probs
            mse_xmis = (variational_params['qy'].repeat((L, 1)).T * mse_xmis.sum(-1)).sum(0).mean()
        kld_xmis = (variational_params['qy'].repeat((L, 1)).T * kld_xmis).sum(0).mean()
    else:
        kld_xmis = torch.tensor([0]).to(data.device)
        mse_xmis = torch.tensor([0]).to(data.device)
        # log_xmis_prior = torch.tensor([0]).to(data.device)
        # recon_data_xmis = compl_data

    if variational_params['qy'] is not None:
        nent_r = variational_params['qy'] * torch.nn.LogSoftmax(1)(variational_params['qy_logit'])
        kld_r = (nent_r.sum(-1) - np.log(1/variational_params['qy'].shape[1])).mean()
        
        kld_z = (variational_params['qy'].repeat((L,1)).T * kld_z).sum(0).sum()
        mse_data = (variational_params['qy'].repeat((L,1)).T * mse_data).sum(0).sum()
        mse_mask = (variational_params['qy'].repeat((L,1)).T * mse_mask).sum(0).sum()

        # recon_data_xobs = torch.einsum('ir, rij -> ij', [variational_params['qy'], recon_data_xobs])
    else:
        kld_r = torch.tensor(0).to(data.device).float()
        mse_data = mse_data.sum()
        mse_mask = mse_mask.sum()
        kld_z = kld_z.sum()

    loss = data_weight * mse_data + mse_mask + mse_xmis + args.z_beta * kld_z + args.r_beta * kld_r + args.xmis_beta * kld_xmis 

    loss_dict = {                 
        mode + ' opt loss': loss,
        mode + ' z KLD':  kld_z,
        mode + ' r KLD':  kld_r,
        mode + ' miss mask MSE': mse_mask,
        mode + ' xobs MSE': mse_data,
        # mode + ' xobs Imputation RMSE': torch.tensor(rmse_loss(recon_data_xobs.detach().cpu().numpy(), compl_data.cpu().numpy(), M_miss.cpu().numpy(), True)).to(data.device),
        # mode + ' xmis Imputation RMSE': torch.tensor(rmse_loss(recon_data_xmis.detach().cpu().numpy(), compl_data.cpu().numpy(), M_miss.cpu().numpy(), True)).to(data.device),
    }

    return loss_dict


def train_VAE(data_train_full, data_test_full, compl_data_train_full, compl_data_test_full, wandb, args, norm_parameters):
        
    args.device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    M_sim_miss_train_full = np.isnan(data_train_full) & ~np.isnan(compl_data_train_full)
    data_train_filled_full = torch.from_numpy(np.nan_to_num(data_train_full.copy(), 0)).to(args.device).float()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data_train_full), torch.from_numpy(compl_data_train_full)), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(data_test_full), torch.from_numpy(compl_data_test_full)), batch_size=args.batch_size, shuffle=True, **kwargs)

    model = model_map[args.model_class](compl_data_train_full.shape[1], args).to(args.device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    norm_type = 'minmax' * args.mnist + 'standard' * (1-args.mnist)
    compl_data_train_full_renorm = renormalization(compl_data_train_full.copy(), norm_parameters, norm_type)

    for epoch in tqdm(range(args.max_epochs)):

        model.train()
        for batch_idx, (data, compl_data) in enumerate(train_loader):
            # data
            data = data.float().to(args.device)
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
            if args.mnist:  
                recon, variational_params, latent_samples = model(data_train_filled_full[:8], torch.from_numpy(M_sim_miss_train_full[:8]))
                save_image_reconstructions_with_mask(recon, compl_data_train_full[:8], M_sim_miss_train_full[:8], variational_params['qy'], epoch, 28, 'images', args.wandb_run_name)
            wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()})
            model.eval()
            with torch.no_grad():
                recon_train, variational_params_train, latent_samples_train = model(data_train_filled_full, torch.tensor(M_sim_miss_train_full).to(args.device))

                train_imputed_xobs_ = renormalization(torch.einsum("ik,kij->ij", [variational_params_train['qy'], recon_train['xobs']]), norm_parameters)
                train_imputed_xobs_ = rounding(train_imputed_xobs_, compl_data_train_full_renorm)
                train_xobs_mis_mse = rmse_loss(train_imputed_xobs_.cpu().numpy().squeeze(), compl_data_train_full_renorm, M_sim_miss_train_full)
                if 'PSMVAE' in args.model_class:
                    train_imputed_xmis_ = renormalization(torch.einsum("ik,kij->ij", [variational_params_train['qy'], recon_train['xmis']]), norm_parameters)
                    train_imputed_xmis_ = rounding(train_imputed_xmis_, compl_data_train_full_renorm)
                    train_xmis_mis_mse = rmse_loss(train_imputed_xmis_.cpu().numpy().squeeze(), compl_data_train_full_renorm, M_sim_miss_train_full)
                else:
                    train_xmis_mis_mse = 0

                wandb.log({'xobs imp rmse': train_xobs_mis_mse, 'xmis imp rmse': train_xmis_mis_mse,})



            #     for batch_idx, (data, compl_data) in enumerate(test_loader):
            #         # data 
            #         data = data.float().to(args.device)
            #         compl_data = compl_data.float().to(args.device)
            #         M_obs = ~data.isnan() & ~compl_data.isnan()
            #         M_miss = data.isnan() & ~compl_data.isnan()
            #         data[data!=data] = 0
            #         # optimization
            #         recon, variational_params, latent_samples = model(data, M_miss)
            #         loss_dict = loss(recon, variational_params, latent_samples, data, compl_data, M_obs, M_miss, args, 'test')
            #         wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()})

    model.eval()
    
    with torch.no_grad():

        if args.model_class == 'PSMVAE_a' or args.model_class == 'PSMVAE_b':
            imp_name = 'xobs' # !!!!!!!!!!!!!!!!!!!!!
        else:
            imp_name = 'xobs'
        # single importance sample
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
            train_imputed_xobs = (train_imputed_1_xobs).clone()
            train_imputed_xmis = (train_imputed_1_xmis).clone()
            train_imputed, train_imputed_1, test_imputed = {
                'xobs': (train_imputed_xobs.cpu().numpy(), train_imputed_1_xobs.cpu().numpy(), test_imputed.cpu().numpy()),
                'xmis': (train_imputed_xmis.cpu().numpy(), train_imputed_1_xmis.cpu().numpy(), test_imputed.cpu().numpy())
            }[imp_name]
        else:
            if not args.mul_imp:
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
            else:
                recon_train, variational_params_train, latent_samples_train = model(data_train_filled_full, torch.tensor(M_sim_miss_train_full).to(args.device), test_mode=True, L=args.num_samples)
                recon_test, variational_params_test, latent_samples_test = model(data_test_filled_full, torch.tensor(M_sim_miss_test_full).to(args.device), test_mode=True, L=args.num_samples)
                if variational_params_train['qy'] == None:
                    recon_train['xobs'] = recon_train['xobs'].repeat((1,1,1,1))
                    variational_params_train['qy'] = torch.ones((data_train_full.shape[0], 1))
                    recon_test['xobs'] = recon_test['xobs'].repeat((1,1,1,1))
                    variational_params_test['qy'] = torch.ones((data_test_full.shape[0], 1))
                train_imputed_xobs = torch.einsum("ik,klij->lij", [variational_params_train['qy'], recon_train['xobs']]) 
                test_imputed = torch.einsum("ik,klij->lij", [variational_params_test['qy'], recon_test[imp_name]])
                if recon_train['xmis'] != None:
                    train_imputed_xmis = torch.einsum("ik,klij->lij", [variational_params_train['qy'], recon_train['xmis']]) 
                else:
                    train_imputed_xmis = torch.from_numpy(data_train_full)
                                
                train_imputed, train_imputed_1, test_imputed = {
                    'xobs': (train_imputed_xobs.cpu().numpy(), train_imputed_1_xobs.cpu().numpy(), test_imputed.cpu().numpy()),
                    'xmis': (train_imputed_xmis.cpu().numpy(), train_imputed_1_xmis.cpu().numpy(), test_imputed.cpu().numpy())
                }[imp_name]

                train_imputed, train_imputed_1, test_imputed = [train_imputed[i] for i in range(args.num_samples)], [train_imputed_1[i] for i in range(args.num_samples)], [test_imputed[i] for i in range(args.num_samples)]

    return(train_imputed, train_imputed_1, test_imputed)
