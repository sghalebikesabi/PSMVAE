import torch
from torch.nn import functional as F


class GMVAE(torch.nn.Module):
    """VAE with GM prior."""
    
    def __init__(self, input_dim, model_params_dict):
        super(GMVAE, self).__init__()

        # input params
        self.m = model_params_dict.miss_mask_training
        self.input_dim = input_dim
        self.input_dim_wm = input_dim + self.m*input_dim
        self.r_cat_dim = model_params_dict.r_cat_dim
        self.z_dim = model_params_dict.z_dim
        self.h_dim = model_params_dict.h_dim

        # q(y|x) y:= r
        self.fc_xobs_h = torch.nn.Linear(self.input_dim_wm, self.h_dim)
        self.fc_hxobs_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_qyl = torch.nn.Linear(self.h_dim, self.r_cat_dim)
        self.fc_qyl_qy = torch.nn.Softmax(1)

        # q(z|x, y) 
        self.fc_xy_h = torch.nn.Linear(self.input_dim_wm + self.r_cat_dim, self.h_dim)
        self.fc_hxy_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_z = torch.nn.Linear(self.h_dim, self.z_dim*2)

        # p(z|y)
        self.fc_y_z = torch.nn.Linear(self.r_cat_dim, self.z_dim*2)
        
        # p(x|z)
        self.fc_z_h = torch.nn.Linear(self.z_dim, self.h_dim)
        self.fc_hz_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_xm = torch.nn.Linear(self.h_dim, self.input_dim_wm)

    def qy_graph(self, xm):
        # q(y|x)
        hxobs = F.relu(self.fc_xobs_h(xm))
        h = F.relu(self.fc_hxobs_h(hxobs))
        qy_logit = self.fc_h_qyl(h)
        qy = self.fc_qyl_qy(qy_logit)
        return qy_logit, qy

    def qz_graph(self, xm, y, test_mode=True, L=1):
        # q(z|x, y)
        xy = torch.cat([xm, y], 1)

        hxy = F.relu(self.fc_xy_h(xy))
        h1 = F.relu(self.fc_hxy_h(hxy))
        z_post = self.fc_h_z(h1)
        z_mu_post, z_logvar_post = torch.split(z_post, self.z_dim, dim=1) 
        z_std_post = torch.sqrt(torch.exp(torch.clamp(z_logvar_post, -15, 15)))

        if test_mode:
            eps = torch.randn((L, z_std_post.shape[0], z_std_post.shape[1])).to(xm.device)
        else:
            eps = torch.randn_like(z_std_post).to(xm.device)

        z = z_mu_post + eps*z_std_post

        if test_mode:
            z = z.reshape((L*z_mu_post.shape[0], z_mu_post.shape[1]))

        return z, z_mu_post, torch.clamp(z_logvar_post, -15, 15)

    def decoder(self, z, y):


        # p(z|y)
        z_prior = self.fc_y_z(y)
        z_mu_prior, z_logvar_prior = torch.split(z_prior, self.z_dim, dim=1) 

        # p(x|z, y) 
        hz = F.relu(self.fc_z_h(z))
        h2 = F.relu(self.fc_hz_h(hz))
        x = self.fc_h_xm(h2)

        recon = {
            'incompl_data': x[:,:self.input_dim],
            'M_sim_miss': torch.sigmoid(x[:,self.input_dim:])
        }
                
        return z_mu_prior, torch.clamp(z_logvar_prior, -15, 15), recon

    def forward(self, x, m, test_mode=False, L=1):
        if self.m:
            xm = torch.cat([x, m], 1)
        else:
            xm = x

        qy_logit, qy = self.qy_graph(xm)
        z, zm, zv, zm_prior, zv_prior, recon = [[None] * self.r_cat_dim for i in range(6)]
        xmis, xmis_mu, xmis_logvar, xmis_mu_prior, xmis_logvar_prior, filled_xm = [[None] * self.r_cat_dim for i in range(6)]
        y_ = torch.zeros([x.shape[0], self.r_cat_dim]).to(x.device)
        for i in range(self.r_cat_dim):
            y = y_ + torch.eye(self.r_cat_dim)[i].to(x.device)
            z[i], zm[i], zv[i] = self.qz_graph(xm, y, test_mode, L)
            zm_prior[i], zv_prior[i], recon[i] = self.decoder(z[i], y)
        
        latent_samples = {
            'z': torch.stack(z),
            'xmis':  None,
        }
        variational_params = {
            'xmis_mu': None,
            'xmis_logvar': None,
            'xmis_mu_prior': None,
            'xmis_logvar_prior': None,
            'z_mu': torch.stack(zm),
            'z_logvar': torch.stack(zv), 
            'z_mu_prior': torch.stack(zm_prior), 
            'z_logvar_prior': torch.stack(zv_prior),
            'qy_logit': qy_logit,
            'qy': qy,
        }
        
        if self.m:
            recon = {
                'xobs': torch.stack([recon[i]['incompl_data'] for i in range(len(recon))]), 
                'xmis': None,
                'M_sim_miss': torch.stack([recon[i]['M_sim_miss'] for i in range(len(recon))])
            }
        else:
            recon = {
                'xobs': torch.stack([recon[i]['incompl_data'] for i in range(len(recon))]), 
                'xmis': None,
                'M_sim_miss': None
            }

        return recon, variational_params, latent_samples
