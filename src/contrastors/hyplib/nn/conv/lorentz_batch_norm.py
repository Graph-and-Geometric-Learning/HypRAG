import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter
from hyplib.manifolds import Lorentz

class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold_in: Lorentz, num_features: int, manifold_out=None):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold_in
        self.c = manifold_in.c
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.manifold_out = manifold_out
        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."
        beta = self.beta
        if self.training:
            mean = self.manifold.lorentzian_centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.lorentzian_centroid(mean)
            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))
            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)
            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.lorentzian_centroid(means, weight=torch.tensor(((1-momentum), momentum), device=means.device))).reshape(self.running_mean.shape))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)
        
        if self.manifold_out is not None:
            output = output * (self.manifold_out.c / self.c).sqrt()

        return output

class LorentzBatchNorm1d(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold_in: Lorentz, num_features: int, manifold_out=None):
        super(LorentzBatchNorm1d, self).__init__(manifold_in, num_features, manifold_out)

    def forward(self, x, momentum=0.1):
        return super(LorentzBatchNorm1d, self).forward(x, momentum)
    
class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold_in: Lorentz, num_channels: int, manifold_out=None):
        super(LorentzBatchNorm2d, self).__init__(manifold_in, num_channels, manifold_out)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d, self).forward(x, momentum)
        x = x.reshape(bs, h, w, c)

        return x