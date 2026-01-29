import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hyplib.nn.linear import LorentzLinear
from hyplib.nn.conv import LResNet

class LorentzPositionEncoding(nn.Module):
    """
    Implements the position embedding layer proposed in Hypformer.
    x: the original Euclidean input, lives in Euclidean space
    y: a vector on the manifold that is some form of x embedded in hyperbolic space.
    
    Parameters:
    ___________
        manifold_in: Lorentz manifold, the manifold for the initial input
        in_channels: dimensionality of input + 1
        out_channels: dimensionality of embedding, same as y
        manifold_out: [Optional] Lorentz manifold, the manifold the embedding lives in. If None, will be the same as manifold_in

    """
        
    def __init__(self, manifold_in, in_channels, out_channels, manifold_out=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.position_embeddings = LorentzLinear(self.manifold_in, self.in_channels, self.out_channels, self.manifold_out)
        self.episilon = torch.nn.Parameter(torch.tensor(1.0))
        self.add = LResNet(manifold_out, weight=self.episilon)

    def forward(self, x, x_emb):
        x = self.position_embeddings(x, x_manifold='euc')
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        print(x_emb.shape)
        x = self.add(x_emb, x)
        return x