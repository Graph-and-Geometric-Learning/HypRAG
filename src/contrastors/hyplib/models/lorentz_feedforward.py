import hyplib.nn as hnn
import math
import torch
from torch import nn
import torch.nn.functional as F

from hyplib.manifolds import Lorentz

class LorentzFeedForward(nn.Module):
    """

    Lorentz Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1: Linear layer for input-to-hidden transformation.
        w2: Linear layer for hidden-to-output transformation.
        w3: Additional linear layer for feature transformation.
    """
    def __init__(self, manifold: Lorentz, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.manifold = manifold
        #TODO: change to these
        # self.w1 = ColumnParallelLinear(dim, inter_dim)
        # self.w2 = RowParallelLinear(inter_dim, dim)
        # self.w3 = ColumnParallelLinear(dim, inter_dim)
        self.c = manifold.c
        self.w1 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)
        self.w2 = hnn.LorentzLinear(self.manifold, inter_dim, dim - 1)
        self.w3 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)

        self.act = hnn.LorentzActivation(self.manifold, F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """

        #TODO: think about this, should we just write a meta function that's like hyp function for space-like dimension???
        x1_time = self.act(self.w1(x))[..., 1:]
        x3_time = self.w3(x)[..., 1:]
        x_space = x1_time * x3_time
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return self.w2(x)