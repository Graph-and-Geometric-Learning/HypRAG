import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from hyplib.manifolds import Lorentz

class LorentzLinear(nn.Module):
    """
    Fully hyperbolic Lorentz layer that allows for varying curvature
    """

    def __init__(self, manifold_in: Lorentz, in_features, out_features, bias=True, manifold_out=None, num_heads=1):
        super().__init__()
        self.in_features = in_features  # time dimension already accounted for
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold_in
        self.c = manifold_in.c
        self.manifold_out = manifold_out
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.num_heads = num_heads

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp', return_space=False):
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)
        if self.num_heads > 1:
            dim_per_head = self.out_features // self.num_heads
            x_space = x_space.reshape(x_space.size(0), x_space.size(1), self.num_heads, dim_per_head)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-8).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x