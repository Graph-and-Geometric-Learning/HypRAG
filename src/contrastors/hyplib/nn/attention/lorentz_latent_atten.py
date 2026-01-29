import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from hyplib.nn.linear import LorentzLinear
from hyplib.nn.conv import LorentzRMSNorm
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from hyplib.manifolds import Lorentz

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

attn_impl = 'naive' #TODO to change

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().contiguous().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class LorentzMLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        manifold (Lorentz): the embedding manifold of the vectors
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, manifold, args):
        #TODO change from args to parameters
        super().__init__()
        self.manifold = manifold
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
            self.wq = LorentzLinear(self.manifold, self.dim, self.n_heads * (self.qk_head_dim - 1)) # Linear operation on the time-like dimensions
        else:
            self.wq_a = LorentzLinear(self.manifold, self.dim, self.q_lora_rank - 1)
            self.q_norm = LorentzRMSNorm(self.manifold, self.q_lora_rank)
            self.wq_b = LorentzLinear(self.manifold, self.q_lora_rank + 1, self.n_heads * (self.qk_head_dim - 1))
        self.wkv_a = LorentzLinear( self.manifold, self.dim, self.kv_lora_rank + self.qk_rope_head_dim - 1)
        self.kv_norm = LorentzRMSNorm(self.manifold, self.kv_lora_rank)
        self.wkv_b = LorentzLinear(self.manifold, self.kv_lora_rank + 1, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim - 1))
        self.wo = LorentzLinear(manifold, self.n_heads * self.v_head_dim, self.dim - 1)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # only cache the time-like dimension
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim - 1), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim - 1), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank - 1), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim - 1), persistent=False)

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], attn_impl='naive'):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q[..., 1:].view(bsz, seqlen, self.n_local_heads, self.qk_head_dim - 1) #time-like
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim - 1], dim=-1) #time-like
        q_pe = apply_rotary_emb(q_pe, freqs_cis) #time-like
        kv = self.wkv_a(x)[..., 1:]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim - 1], dim=-1) #time-like
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) #time-like
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1) #time-like
            kv = self.wkv_b(self.kv_norm(self.project(kv)))[..., 1:] #time-like
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim - 1)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim - 1], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            
            # MLA based on hyperbolic distance
            qs = self.project(q)
            # ks = self.project(self.k_cache[:bsz, :end_pos])
            ks = self.project(k)
            scores = 2 * self.manifold.c + 2 * self.manifold.cinner(qs.transpose(1, 2), ks.transpose(1, 2)) * self.softmax_scale # [B, S, N, N]
            if mask is not None:
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
        else:
            # doesn't work yet..., need rethinking
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            if mask is not None:
                scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            # x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            # vs = self.project(self.v_cache[:bsz, :end_pos])
            vs = self.project(v)
            x = self.manifold.lorentzian_centroid(vs.transpose(1, 2), scores).transpose(1, 2) #[B, S, H, N]
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
            x_space = x.flatten(2)
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
            x = torch.cat([x_time, x_space], dim=-1)
        x = self.wo(x.flatten(2))
        return x