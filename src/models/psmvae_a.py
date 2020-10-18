import torch
from torch import nn
from torch.nn import functional as F


class Model(torch.nn.Module):
    
    def __init__(self, input_dim, model_params_dict):
        super(Model, self).__init__()

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

        # q(xmis|x, y) 
        self.fc_xobsy_h = torch.nn.Linear(self.input_dim_wm + self.z_dim + self.r_cat_dim, self.h_dim)
        self.fc_hxobsy_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_xmis = torch.nn.Linear(self.h_dim, self.input_dim*2)

        # p(xmis|y) 
        # self.fc_y_xmis = torch.nn.Linear(self.r_cat_dim, self.input_dim)

        # q(z|x, y) 
        self.fc_xy_h = torch.nn.Linear(self.input_dim_wm + self.r_cat_dim, self.h_dim)
        self.fc_hxy_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_z = torch.nn.Linear(self.h_dim, self.z_dim*2)

        # p(z|y)
        self.fc_y_z = torch.nn.Linear(self.r_cat_dim, self.z_dim*2)
        
        # p(x|z, r)
        self.fc_z_h = torch.nn.Linear(self.z_dim + self.r_cat_dim, self.h_dim)
        self.fc_hz_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_xm = torch.nn.Linear(self.h_dim, self.input_dim + self.input_dim)

        self.fc_x_m = torch.nn.Linear(self.input_dim*2 + self.r_cat_dim, self.input_dim)
        self.W = torch.nn.ParameterList([nn.Parameter(torch.zeros((self.input_dim, 1), device=model_params_dict.device)) for i in range(self.r_cat_dim)])
        self.b = torch.nn.ParameterList([nn.Parameter(torch.zeros((1, self.input_dim), device=model_params_dict.device)) for i in range(self.r_cat_dim)])
        for i in range(self.r_cat_dim):
            torch.nn.init.xavier_normal_(self.W[i])

    def qy_graph(self, xm):
        # q(y|x)
        hxobs = F.relu(self.fc_xobs_h(xm))
        h = F.relu(self.fc_hxobs_h(hxobs))
        qy_logit = self.fc_h_qyl(h)
        qy = self.fc_qyl_qy(qy_logit)
        return qy_logit, qy

    def qxmis_graph(self, xm, z, y, test_mode, L=1):
        # q(x|z, y, x)
        xobsy = torch.cat([xm, z, y], 1)

        hxobsy = F.relu(self.fc_xobsy_h(xobsy))
        hobs = F.relu(self.fc_hxobsy_h(hxobsy))
        xmis_post = self.fc_h_xmis(hobs)
        xmis_mu_post, xmis_logvar_post = torch.split(xmis_post, self.input_dim, dim=1) 
        xmis_std_post = torch.sqrt(torch.exp(torch.clamp(xmis_logvar_post, -15, 15)))

        eps = torch.randn_like(xmis_std_post).to(xm.device)
        xmis = xmis_mu_post + eps*xmis_std_post
        
        return xmis, xmis_mu_post, torch.clamp(xmis_logvar_post, -15, 15) #prevent numerical inaccuracies

    def qz_graph(self, xm, y, test_mode, L=1):
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

    def decoder(self, z, y, m, i, test_mode, L=1):

        # p(xmis|y) 
        # xmis_mu_prior = self.fc_y_xmis(y)
        # xmis_logvar_prior = torch.zeros_like(xmis_mu_prior).to(z.device)

        # p(z|y)
        z_prior = self.fc_y_z(y)
        z_mu_prior, z_logvar_prior = torch.split(z_prior, self.z_dim, dim=1) 

        # p(x|z, y) 
        if test_mode:
            hz = F.relu(self.fc_z_h(torch.cat([z, y.repeat((L, 1))], 1)))
        else:
            hz = F.relu(self.fc_z_h(torch.cat([z, y], 1)))
        h2 = F.relu(self.fc_hz_h(hz))
        x = self.fc_h_xm(h2)

        m_input = x[:,:self.input_dim] * (1-m.float()) + x[:,self.input_dim:self.input_dim*2] * m

        recon = {
            'xobs': x[:,:self.input_dim],
            'xmis': x[:,self.input_dim:self.input_dim*2],
            # 'M_sim_miss': torch.sigmoid(self.fc_x_m(torch.cat([x[:,:self.input_dim], x[:, self.input_dim:self.input_dim*2], y.repeat((L, 1))], 1))),
            'M_sim_miss': torch.sigmoid(torch.einsum("ij, jk -> ij", [m_input, torch.nn.functional.softplus(self.W[i])]) + self.b[i]),
        }
                
        return z_mu_prior, torch.clamp(z_logvar_prior, -15, 15), recon #xmis_mu_prior, torch.clamp(xmis_logvar_prior, -15, 15), 

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
            if not test_mode:
                z[i], zm[i], zv[i] = self.qz_graph(xm, y, test_mode, L)
                xmis[i], xmis_mu[i], xmis_logvar[i] = self.qxmis_graph(xm, z[i], y, test_mode, L)
            else:
                z[i], zm[i], zv[i] = self.qz_graph(xm, y, test_mode, L)
                xmis[i], xmis_mu[i], xmis_logvar[i] = self.qxmis_graph(xm.repeat((L, 1)), z[i], y.repeat((L, 1)), test_mode, L)
            zm_prior[i], zv_prior[i], recon[i] = self.decoder(z[i], y, m, i, test_mode, L)
            # zm_prior[i], zv_prior[i], xmis_mu_prior[i], xmis_logvar_prior[i], recon[i] = self.decoder(z[i], y, test_mode, L)
        
        latent_samples = {
            'z': torch.stack(z),
            'xmis':  torch.stack(xmis),
        }
        variational_params = {
            'xmis_mu': torch.stack(xmis_mu),
            'xmis_logvar': torch.stack(xmis_logvar),
            'xmis_mu_prior': None, #torch.stack(xmis_mu_prior),
            'xmis_logvar_prior': None, #torch.stack(xmis_logvar_prior),
            'z_mu': torch.stack(zm),
            'z_logvar': torch.stack(zv), 
            'z_mu_prior': torch.stack(zm_prior), 
            'z_logvar_prior': torch.stack(zv_prior),
            'qy_logit': qy_logit,
            'qy': qy,
        }
        
        recon = {
            'xobs': torch.stack([recon[i]['xobs'] for i in range(len(recon))]), 
            'xmis': torch.stack([recon[i]['xmis'] for i in range(len(recon))]),
            'M_sim_miss': torch.stack([recon[i]['M_sim_miss'] for i in range(len(recon))]),
        }

        return recon, variational_params, latent_samples
