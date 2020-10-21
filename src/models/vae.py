import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, num_in, model_params_dict):
        super(VAE, self).__init__()

        self.device = model_params_dict.device
        self.num_samples = model_params_dict.num_samples

        self.fc1 = nn.Linear(num_in, model_params_dict.h_dim)
        self.fc2 = nn.Linear(model_params_dict.h_dim, model_params_dict.h_dim)
        self.fc21 = nn.Linear(model_params_dict.h_dim, model_params_dict.z_dim)
        self.fc22 = nn.Linear(model_params_dict.h_dim, model_params_dict.z_dim)
        self.fc3 = nn.Linear(model_params_dict.z_dim, model_params_dict.h_dim)
        self.fc3b = nn.Linear(model_params_dict.h_dim, model_params_dict.h_dim)
        self.fc4 = nn.Linear(model_params_dict.h_dim, num_in)
        self.mnist = model_params_dict.mnist

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar, test_mode=True, L=1):
        std = torch.exp(0.5*logvar)
        
        if test_mode:
          eps = torch.randn((L, std.shape[0], std.shape[1]))
        else:
          eps = torch.randn_like(std).to(self.device)
        
        z = mu + eps*std

        if test_mode:
          z = z.reshape((L*mu.shape[0], mu.shape[1]))

        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3b = F.relu(self.fc3b(h3))
        h4 = self.fc4(h3b)
        if self.mnist:
          h4 = torch.sigmoid(h4)
        return h4

    def forward(self, x, m, test_mode=False, L=1):
      if self.mnist:
        x = x.view(-1, 784)
      mu, logvar = self.encode(x)
      z = self.reparameterize(mu, logvar, test_mode, L)
      recon = {'xobs': self.decode(z), 'xmis': None, 'M_sim_miss': None}
      variational_params = {
        'z_mu': mu, 
        'z_logvar': logvar, 
        'z_mu_prior': torch.zeros_like(mu).to(self.device), 
        'z_logvar_prior': torch.zeros_like(logvar).to(self.device), 
        'qy': None,
        'xmis':  None,
        'xmis_mu': None,
        'xmis_logvar': None,
        'xmis_mu_prior': None,
        'xmis_logvar_prior': None,
      }
      latent_samples = {'z': z}
      return recon, variational_params, latent_samples
