import torch
import torch.nn as nn
from rgcn_model import RGCN

class DSG(nn.Module):
    def __init__(self, args):
        super(DSG, self).__init__()
        self.enc_layers = RGCN(args)

    def encoder(self, batch):
        h = self.enc_layers(batch)
        return h

    def sampling(self, mean, std):
        eps = torch.randn(mean.size(0), std.size(0)).cuda()
        z_sampled = mean + eps * std
        return z_sampled

    def forward(self, sup_g_bidir):
        # domain shifts with uncertainty
        eps = 1e-5
        h = self.encoder(sup_g_bidir)

        mean = torch.mean(h, dim=1)
        std = torch.std(h, dim=1, correction=0)
        mean = mean.unsqueeze(1).expand_as(h)
        std = std.unsqueeze(1).expand_as(h)

        std_mu = torch.std(mean, dim=0, keepdim=True, correction=0)
        std_var = torch.std(std, dim=0, keepdim=True, correction=0)

        beta = self.sampling(mean, std_mu)
        gam = self.sampling(std, std_var)

        z = beta + gam * ((h-mean) / (std+eps))

        sup_g_bidir.ndata['g'] = z
        sup_g_bidir.ndata['h'] = h

        return h, z
