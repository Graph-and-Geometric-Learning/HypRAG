import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from hyplib.manifolds import Lorentz


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """
    def __init__(
            self, 
            manifold: Lorentz, 
            num_features: int, 
            num_classes: int,
            ignore_index: int = -1,
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes, num_features-1), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1/self.manifold.k.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK*self.a)*norm_z)
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(-w_t**2+torch.norm(w_s, dim=-1)**2)
        alpha = -w_t*x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK*self.a)*torch.inner(x.narrow(-1, 1, x.shape[-1]-1), self.z))

        d = self.manifold.k.sqrt()*torch.abs(torch.asinh(sqrt_mK*alpha/beta))  # Distance to hyperplane
        logits = torch.sign(alpha)*beta*d

        return logits
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)
