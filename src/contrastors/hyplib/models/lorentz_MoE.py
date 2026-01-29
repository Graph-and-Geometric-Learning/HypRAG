
import hyplib.nn as hnn
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from hyplib.manifolds import Lorentz

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

class LorentzGate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    This can be seen as the operation on the time-like dimension
    TODO: hyperbolic version??? Not sure how yet...
    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, manifold: Lorentz, args):
        """
        Initializes the Gate module.

        Args:
            args (DeepSeekModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # scores = linear(x, self.weight) TODO: make this possibke

        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

class LorentzMLP(nn.Module):
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

class LorentzExpert(nn.Module):
    """
    Expert layer for Lorentz Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, manifold, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.c = manifold.c
        self.manifold =manifold
        self.w1 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)
        self.w2 = hnn.LorentzLinear(self.manifold, inter_dim, dim - 1)
        self.w3 = hnn.LorentzLinear(self.manifold, dim, inter_dim - 1)

        self.act = hnn.LorentzActivation(self.manifold, F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        x1_time = self.act(self.w1(x))[..., 1:]
        x3_time = self.w3(x)[..., 1:]
        x_space = x1_time * x3_time
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return self.w2(x)

class LorentzMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, manifold: Lorentz, args):
        """
        Initializes the MoE module.

        Args:
            args (DeepSeekModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.manifold = manifold
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = LorentzGate(manifold, args)
        self.experts = nn.ModuleList([LorentzExpert(self.manifold, args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = LorentzMLP(self.manifold, args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.add_experts = hnn.LResNet(self.manifold, use_scale=True, scale=10.0)

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = self.project(torch.zeros_like(x[..., 1:]))
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] = hnn.LResNet(self.manifold, weight=weights[idx, top, None])(y[idx], expert(x[idx]))
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        out = self.add_experts(y, z)
        return out.view(shape)