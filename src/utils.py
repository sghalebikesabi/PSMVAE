import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

def make_deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def impute(model, data, args, kwargs):
    data_loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=1, shuffle=False, **kwargs)
    xobs_imputed_full = torch.zeros(data.shape).to(args.device)
    xmis_imputed_full = torch.zeros(data.shape).to(args.device)
    for batch_idx, (data,) in enumerate(data_loader):
        data_filled = data.float().to(args.device)
        data_filled[data_filled!=data_filled] = 0
        M_miss = data.isnan().to(args.device).float()
        M_obs = 1. - M_miss
        recon, variational_params, latent_samples = model(data_filled, M_miss, test_mode=True, L=args.num_samples)
        variational_params['py'] = torch.ones_like(variational_params['qy']).to(args.device)/variational_params['qy'].shape[1]

        log_p_xobs_given_z_r =  M_obs*log_normal(data_filled, recon['xobs'], torch.tensor([0.25]).to(args.device)) # ! 0.25
        if 'PSMVAE' in args.model_class:
          log_p_xmis_given_z_r =  log_normal(latent_samples['xmis']*M_miss, recon['xmis']*M_miss, torch.tensor([0.25]).to(args.device)) 
          # log_p_xmis_given_z_r = 0 # log_normal(latent_samples['xmis']*M_miss, recon['xmis']*M_miss, torch.tensor([0.5])) 
          log_p_xmis_given_z_r += log_normal(latent_samples['xmis']*M_obs, recon['xmis']*M_obs, torch.tensor([0.25]).to(args.device)) * args.pi # ! 0.25
          log_p_xmis_given_z_r += log_normal(data_filled*M_obs, recon['xmis']*M_obs, torch.tensor([0.25]).to(args.device)) * (1 - args.pi) # ! 0.25
        else:
          log_p_xmis_given_z_r = torch.zeros_like(log_p_xobs_given_z_r)

        
        log_p_z_given_r = log_normal(latent_samples['z'], 
                                      variational_params['z_mu'].reshape(args.r_cat_dim, 1, 1,  args.z_dim).repeat(1, args.num_samples, 1, 1),
                                      torch.exp(variational_params['z_logvar'].reshape(args.r_cat_dim, 1, 1,  args.z_dim).repeat(1, args.num_samples, 1, 1)))

        # variational_params['z_mu_prior'].shape
        # torch.Size([10, 1, 20])
        # latent_samples['z'].shape
        # torch.Size([10, 20, 1, 20])

        log_p_r = variational_params['py']

        log_q_z_given_r_xobs = log_normal(latent_samples['z'], 
                                            variational_params['z_mu'].reshape(args.r_cat_dim, 1, 1,  args.z_dim).repeat(1, args.num_samples, 1, 1), 
                                            torch.exp(variational_params['z_logvar'].reshape(args.r_cat_dim, 1, 1,  args.z_dim).repeat(1, args.num_samples, 1, 1)))
        
        log_q_r_xobs_m = variational_params['qy']

        imp_weights = log_p_z_given_r.sum(-1).squeeze() + log_p_r.T.repeat((1, args.num_samples)) - log_q_z_given_r_xobs.sum(-1).squeeze() - log_q_r_xobs_m.T.repeat((1, args.num_samples)).squeeze()
        imp_weights_xobs = torch.nn.functional.softmax((imp_weights + log_p_xobs_given_z_r.sum(-1).squeeze()).reshape(args.num_samples*args.r_cat_dim), 0) 
        imp_weights_xmis = torch.nn.functional.softmax((imp_weights + log_p_xmis_given_z_r.sum(-1).squeeze()).reshape(args.num_samples*args.r_cat_dim), 0) 
        xobs_imputed_full[batch_idx, :] = torch.einsum('k,kj->j', imp_weights_xobs.float(), recon['xobs'].reshape(args.num_samples*args.r_cat_dim, recon['xobs'].shape[-1]))
        if 'PSMVAE' in args.model_class:
          xmis_imputed_full[batch_idx, :] = torch.einsum('k,kj->j', imp_weights_xmis.float(), recon['xmis'].reshape(args.num_samples*args.r_cat_dim, recon['xmis'].shape[-1]))
        else:
          xmis_imputed_full = xobs_imputed_full

    return(xobs_imputed_full, xmis_imputed_full)


def rmse_loss(imputed_data, ori_data, data_m, norm=True):
    '''Compute RMSE loss between ori_data and imputed_data for missing data

    Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness, 1 if observation is missing

    Returns:
    - rmse: Root Mean Squared Error
    '''

    if norm:
      ori_data, norm_parameters = normalization(ori_data.copy(), None, 'minmax')
      imputed_data, _ = normalization(imputed_data.copy(), norm_parameters, 'minmax')

    # Only for missing values
    nominator = np.sum((((data_m) * np.nan_to_num(ori_data) - (data_m) * imputed_data)**2))
    denominator = np.sum(data_m) + 1e-7

    rmse = np.sqrt(nominator/float(denominator))

    return rmse


def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = torch.add(var, eps, name='clipped_var')
    return -0.5 * (np.log(2 * math.pi) + torch.log(var) + torch.square(x - mu) / var)


def normal_KLD(z, zm, zv, zm_prior, zv_prior):
    return(log_normal(z, zm, torch.exp(zv)) - log_normal(z, zm_prior, torch.exp(zv_prior)))


def cluster_acc(qy_logit, targets):
    with torch.no_grad():
        cat_pred = qy_logit.argmax(1)
        real_pred = torch.zeros_like(cat_pred).to(cat_pred.device)
        for cat in range(qy_logit.shape[1]):
            idx = (cat_pred == cat)
            lab = targets[idx]
            if len(lab) == 0:
                continue
            real_pred[cat_pred == cat] = lab.mode()[0] 
    acc = torch.mean(real_pred == targets)  
    return(acc)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def normalization (data, norm_params = None, norm_type='standard'):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data

  if norm_type == 'minmax':

    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    if norm_params == None:
      # For each dimension
      for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:,i])
        max_val[i] = np.nanmax(norm_data[:,i])
        if data.shape[1]==784:
          min_val[i] = 0
          max_val[i] = 255
        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
    else:
        for i in range(dim):
          norm_data[:,i] = norm_data[:,i] - norm_params['min_val'][i]
          norm_data[:,i] = norm_data[:,i] / (norm_params['max_val'][i] + 1e-6)   


    # Return norm_parameters for renormalization
    norm_params = {'min_val': min_val,
                      'max_val': max_val}
  
  elif norm_type == 'standard':

    if norm_params == None:
      mu = np.nanmean(norm_data, axis=0)
      std = np.nanstd(norm_data, axis=0) + 1e-6

      norm_params = {'mu': mu,
                    'std': std}

    norm_data = (norm_data- norm_params['mu'])/norm_params['std']
      
  return norm_data, norm_params


def renormalization (norm_data, norm_parameters, norm_type='standard'):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  renorm_data = norm_data

  if norm_type == 'minmax':
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
      
    for i in range(dim):
      renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
      renorm_data[:,i] = renorm_data[:,i] + min_val[i]

  elif norm_type == 'standard':
      renorm_data *= norm_parameters['std']
      renorm_data += norm_parameters['mu']

  return renorm_data


def save_image_reconstructions(imputed, X, M, image_dim_0, results_path, tag):
    # sample images 
    input_dim = X.shape[1]
    n = min(X.shape[0], 8)
    recon_sample_X = imputed[M.sum(1)>0][:n]
    sample_X = X[M.sum(1)>0][:n]
    sample_M = M[M.sum(1)>0][:n]
    #induce red missingness
    sample_X_with_red_miss, _ = induce_red_missingness(sample_X, sample_M, image_dim_0)
    RGB_sample_image = np.repeat(np.reshape(recon_sample_X, (recon_sample_X.shape[0], 1, image_dim_0, -1)), repeats = [3], axis=1)
    comparison = torch.cat([sample_X_with_red_miss.float(), torch.from_numpy(RGB_sample_image)])
    save_image(comparison.cpu(), results_path + '/reconstruction_' + str(tag) + '.png', nrow=n)


def save_image_reconstructions_with_mask(recon_batch, X, M, qy, epoch, image_dim_0, results_path, tag):
    # sample images 
    input_dim = X.shape[1]
    n = min(X.shape[0], 8)
    if qy != None:
        classes_train = qy.argmax(1)
        recon_sample_X = torch.einsum("ik, kij -> ij", [qy[:n], recon_batch['xobs'][:, :n]])
        # recon_sample_X = torch.stack([recon_batch['xobs'][classes_train[i], i, :] for i in range(n)], 0)
    else:
        recon_sample_X = recon_batch['xobs'][:, :n]

    try:
        recon_sample_M = torch.einsum("ik, kij -> ij", [qy[:n], recon_batch['M_sim_miss'][:, :n]])
        recon_sample_M = np.round(recon_sample_M.detach().cpu().numpy())
    except AttributeError:
        recon_sample_M = np.zeros(recon_sample_X.shape)
    sample_X = X[:n]
    sample_M = M[:n]
    #induce red missingness
    sample_X_with_red_miss, _ = induce_red_missingness(sample_X, sample_M, image_dim_0)
    recon_sample_X_with_red_miss, RGB_recon_sample_X = induce_red_missingness(recon_sample_X, recon_sample_M, image_dim_0)
    comparison = torch.cat([sample_X_with_red_miss.float(), recon_sample_X_with_red_miss.float(), RGB_recon_sample_X])
    save_image(comparison.cpu(), results_path + '/reconstruction_' + str(tag) + '_' + str(epoch)  + '.png', nrow=n)


def induce_red_missingness(sample_X, sample_M, image_dim_0):
    """Function that induces red pixels in an image if the pixel values are missing.

    Args:
        sample_X: sample images saved as numpy array or pytorch tensor of dimensions [number of images, imagedim1* imagedim2].
        sample_M: sample missingness masks of same structure as 'sample_X'.
        image_dim_0: rows of image.
    
    Returns:
        A torch tensor of the sample images with dimensions [number of images, 3, imagedim1, imagedim2]
    """
    # ensure input is numpy
    if torch.is_tensor(sample_X):
        sample_X = sample_X.detach().numpy()
    if torch.is_tensor(sample_M):
        sample_M = sample_M.detach().numpy()
    
    # reshape input
    reshaped_sample_X = np.reshape(sample_X, (sample_X.shape[0], 1, image_dim_0, -1))
    reshaped_sample_M = np.reshape(sample_M, (sample_X.shape[0], 1, image_dim_0, -1))

    # turn to RGB scale
    RGB_sample_image = np.repeat(reshaped_sample_X, repeats = [3], axis=1)
    RGB_sample_M = np.repeat(reshaped_sample_M, repeats=[3], axis=1)
    
    # add red color
    RGB_sample_image_with_red_miss = RGB_sample_image.copy()
    for i in range(reshaped_sample_X.shape[0]):
        for j in range(reshaped_sample_X.shape[2]):
            for k in range(reshaped_sample_X.shape[3]):
                if RGB_sample_M[i,0,j,k]:
                    # RGB_sample_image_with_red_miss[i,:,j,k] = [255*RGB_sample_M[i,0,j,k], 0, 0]
                    RGB_sample_image_with_red_miss[i,:,j,k] = [255, 0, 0]

    return(torch.tensor(RGB_sample_image_with_red_miss), torch.tensor(RGB_sample_image))


def generate_synthetic_data(n=5000, rho=0.1, miss_ratio=0.2, train_pct=0.8, val_pct=0.1, seed=0):
    import numpy as np
    import os
    import pandas as pd
    import random

    random.seed(seed)
    mean = [2, -2, 0]  
    rho = 0.1
    cov = [[1, rho, rho], [rho, 1, 0], [rho, 0, 1]]

    data = np.random.multivariate_normal(mean, cov, (n))
    observedi = np.random.choice([0, 1], n, p = [miss_ratio, 1-miss_ratio])
    data[:, 1][observedi] = data[:, 2][observedi] # column 2 is observed data, column 1 is miss data
    new_data = pd.DataFrame(np.delete(data, 2, axis = 1))
    miss_data = new_data.copy()
    miss_data[(1-observedi)>0, 1] = np.nan
    M = pd.DataFrame(np.isnan(miss_data))

    train_idx = int(train_pct*n)
    val_idx = int((train_pct+val_pct)*n)
    random_permute = np.random.RandomState(seed=seed).permutation(len(data))
    M_train, M_val, M_test = M.iloc[random_permute[:train_idx]], M.iloc[random_permute[train_idx:val_idx]], M.iloc[random_permute[val_idx:]]  
    data_train, data_val, data_test = new_data.iloc[random_permute[:train_idx]], new_data.iloc[random_permute[train_idx:val_idx]], new_data.iloc[random_permute[val_idx:]]  

    data_file = "data/synth"
    data_name = f"n_{n}_m_2_seed_{seed}"
    miss_file_name = f"n_{n}_m_2_missratio_20_seed_{seed}"

    data_train.to_csv(os.path.join(data_file, f"{data_name}.train"), header=False, index=False)
    data_val.to_csv(os.path.join(data_file, f"{data_name}.val"), header=False, index=False)
    data_test.to_csv(os.path.join(data_file, f"{data_name}.test"), header=False, index=False)

    M_train.to_csv(os.path.join(data_file, "miss_data", miss_file_name + ".train"), header=False, index=False)
    M_val.to_csv(os.path.join(data_file, "miss_data", miss_file_name + ".val"), header=False, index=False)
    M_test.to_csv(os.path.join(data_file, "miss_data", miss_file_name + ".test"), header=False, index=False)
