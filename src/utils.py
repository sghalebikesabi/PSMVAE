import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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

        log_p_xmis_given_z_r =  log_normal(latent_samples['xmis']*M_miss, recon['xmis']*M_miss, torch.tensor([0.5])) 
        # log_p_xmis_given_z_r = 0 # log_normal(latent_samples['xmis']*M_miss, recon['xmis']*M_miss, torch.tensor([0.5])) 
        log_p_xmis_given_z_r += log_normal(latent_samples['xmis']*M_obs, recon['xmis']*M_obs, torch.tensor([0.25])) * args.pi
        log_p_xmis_given_z_r += log_normal(data_filled*M_obs, recon['xmis']*M_obs, torch.tensor([0.25])) * (1 - args.pi)
        log_p_xobs_given_z_r =  M_obs*log_normal(data_filled, recon['xobs'], torch.tensor([0.25]))
        
        log_p_z_given_r = log_normal(latent_samples['z'], variational_params['z_mu_prior'], torch.exp(variational_params['z_logvar_prior']))

        log_p_r = variational_params['py']

        log_q_z_given_r_xobs = log_normal(latent_samples['z'], variational_params['z_mu'], torch.exp(variational_params['z_logvar']))
        
        log_q_r_xobs_m = variational_params['qy']

        imp_weights = log_p_z_given_r.sum(-1) + log_p_r.T.repeat((1, args.num_samples)) - log_q_z_given_r_xobs.sum(-1) - log_q_r_xobs_m.T.repeat((1, args.num_samples))
        imp_weights_xobs = torch.nn.functional.softmax((imp_weights + log_p_xobs_given_z_r.sum(-1)).reshape(args.num_samples*args.r_cat_dim), 0) 
        imp_weights_xmis = torch.nn.functional.softmax((imp_weights + log_p_xmis_given_z_r.sum(-1)).reshape(args.num_samples*args.r_cat_dim), 0) 
        xobs_imputed_full[batch_idx, :] = torch.einsum('k,kj->j', imp_weights_xobs.float(), recon['xobs'].reshape(args.num_samples*args.r_cat_dim, recon['xobs'].shape[-1]))
        xmis_imputed_full[batch_idx, :] = torch.einsum('k,kj->j', imp_weights_xmis.float(), recon['xmis'].reshape(args.num_samples*args.r_cat_dim, recon['xmis'].shape[-1]))

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
      ori_data, norm_parameters = normalization(ori_data, None, 'minmax')
      imputed_data, _ = normalization(imputed_data, norm_parameters, 'minmax')

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
        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        max_val[i] = np.nanmax(norm_data[:,i])
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

