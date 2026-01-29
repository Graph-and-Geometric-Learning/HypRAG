import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hyplib.nn.conv import LorentzConv2d

class LorentzPatchEmbedding(nn.Module):
    """
    Implements Lorentz patch embedding using a Lorentz 2D convolutional layer.
    """
        
    def __init__(self, manifold_in, image_size, patch_size, in_channel, out_channel, manifold_out=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = LorentzConv2d(manifold_in=manifold_in, in_channels=in_channel, out_channels=out_channel, kernel_size=self.patch_size, stride=self.patch_size, manifold_out=manifold_out)

    def forward(self, x):
        # make x channel-last
        # x = x.permute(0, 2, 3, 1)
        x = self.projection(x) # batch_size * num_patches_x * num_patches_y * channels
        x = x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        return x