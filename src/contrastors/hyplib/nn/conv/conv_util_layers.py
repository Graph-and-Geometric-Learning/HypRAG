'''
Hyperbolic Layers that makes up hyperolic convolution
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LorentzLayerNorm(nn.Module):
    def __init__(self, manifold_in, in_features, manifold_out=None):
        super(LorentzLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold_in
        self.c = manifold_in.c
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, space_only = False, return_space = False):
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        x_space = self.layer(x_space)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzNormalization(nn.Module):
    def __init__(self, manifold_in, manifold_out=None):
        super(LorentzNormalization, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.c = manifold_in.c

    def forward(self, x, norm_factor=None):
        x_space = x[..., 1:]
        if norm_factor is not None:
            x_space = x_space * norm_factor
        else:
            x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzActivation(nn.Module):
    def __init__(self, manifold_in, activation, manifold_out=None):
        super(LorentzActivation, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.activation = activation
        self.c = manifold_in.c

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
        return x

class LorentzDropout(nn.Module):
    def __init__(self, manifold_in, dropout, manifold_out=None):
        super(LorentzDropout, self).__init__()
        self.manifold = manifold_in
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)
        self.c = manifold_in.c

    def forward(self, x, training=False):
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
        return x
    
class LResNet(nn.Module):
    def __init__(self, manifold_in, weight=None, batch_size=None, use_scale=False, scale=None, manifold_out=None):
        super(LResNet, self).__init__()
        self.manifold = manifold_in
        if weight is not None:
            self.w_y = weight
        else:
            # using learnable weights
            if batch_size:
                # use separate weight for each vector
                self.w_y = nn.Parameter(torch.ones((batch_size, 1)))
            else:
                self.w_y = nn.Parameter(torch.tensor(1.0))
        self.scale = None
        if use_scale:
            if scale:
                self.scale = scale
                self.learned_scale = False
            else:
                self.scale = nn.Parameter(torch.tensor(3.0))
                self.learned_scale = True
        self.c = manifold_in.c
        self.manifold_out = manifold_out

    def forward(self, x, y):
        ave = x + y * self.w_y
        denom = (-self.manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-6).sqrt()
        x = self.c.sqrt() * ave / denom
        if self.scale:
            if self.learned_scale:
                x_space = self.scale.exp().clamp_max(30.0) * x[..., 1:]
            else:
                x_space = self.scale * x[..., 1:]
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()

        return x

class LorentzRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, manifold_in, dim: int, eps: float = 1e-6, manifold_out=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.c = manifold_in.c

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        x_space = x[..., 1:]
        normed_space = F.rms_norm(x_space, (self.dim,), self.weight, self.eps)
        x_time = ((normed_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, normed_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x
