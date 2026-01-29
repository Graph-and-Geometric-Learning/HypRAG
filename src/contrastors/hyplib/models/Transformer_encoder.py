from __future__ import annotations
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from hyplib.optimizers import Optimizer, LR_Scheduler
import numpy as np
import random
import time
import math
import logging
import torch
import torch.nn as nn
import hyplib.nn as hnn
from .lorentz_feedforward import LorentzFeedForward   

import os
from hyplib.manifolds import Lorentz
import re
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import (
    create_block_mask,
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# flex_attention_compiled = torch.compile(flex_attention, fullgraph=True, mode='max-autotune', dynamic=False)
flex_attention_compiled = torch.compile(flex_attention)
# flex_attention_compiled = flex_attention

kernel_options = {}

# check CUDA type, if it is 'L40S', then add kernel options for L40S
if "L40S" in torch.cuda.get_device_properties(0).name:
    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    }
elif "H100" in torch.cuda.get_device_properties(0).name or "H200" in torch.cuda.get_device_properties(0).name:
    kernel_options = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "num_stages": 2,
        "FORCE_USE_FLEX_ATTENTION": True,  # TODO inspect flex_decode
    }


# class _LTransformerBlock(nn.Module):
#     """
#     Single transformer block comprising multi-head self-attention and MLP, in hyperbolic space. Both
#     modules are preceeding by hyperbolic layer normalization. This module is the hyperbolic person of PyTorch's
#     builtin module `TransformerEncoderLayer` with arguments as (`norm_first=True, dropout=0, activation="gelu"`).
#     """

#     def __init__(self, manifold, d_model: int, n_head: int):
#         super().__init__()
#         dim_per_head = d_model // n_head
#         self.manifold = manifold
#         self.attn = hnn.LorentzMultiheadAttention(manifold, dim_per_head, dim_per_head, n_head, attention_type='full', trans_heads_concat=True)
#         # self.ln_1 = hnn.LorentzLayerNorm(manifold, d_model - 1)
#         self.DyT1 = DyT(d_model-1)
#         self.DyT2 = DyT(d_model-1)
#         # self.DyT3 = DyT(d_model-1)
#         self.mlp = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("c_fc", hnn.LorentzLinear(manifold, d_model, d_model * 4 - 1)),
#                     ("gelu", hnn.LorentzActivation(manifold, activation=nn.GELU())),
#                     ("c_proj", hnn.LorentzLinear(manifold, d_model * 4, d_model - 1)),
#                 ]
#             )
#         )
#         self.scale = nn.Parameter(torch.tensor(3.0))
#         # self.ln_2 = hnn.LorentzLayerNorm(manifold, d_model - 1)
#         self.res1 = hnn.LResNet(manifold, use_scale=True, scale=2*math.sqrt(d_model)) # Try using 2 * sqrt of the dimension. Such large number is not necessary for small model.
#         self.res2 = hnn.LResNet(manifold, use_scale=True, scale=2*math.sqrt(d_model))
#         self.activation = nn.Tanh()

#     def project(self, x):
#         space_norm2 = (x ** 2).sum(dim=-1, keepdim=True)
#         time_sq = space_norm2 +self.manifold.c
#         time_sq = torch.clamp(time_sq, min=1e-4)
#         x_time = torch.sqrt(time_sq)
#         return torch.cat([x_time, x], dim=-1)

#     def forward(self, x, attn_mask=None):
#         # DyT1 + project
#         dyt1_out = self.DyT1(x[...,1:])
#         # assert not torch.isnan(dyt1_out).any(), "NaN detected after DyT1"
        
#         lx = self.project(dyt1_out)
#         # assert not torch.isnan(lx).any(), "NaN detected after project(DyT1)"

#         # Attention
#         ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask)
#         # assert not torch.isnan(ax).any(), "NaN detected after attn"

#         # Residual 1
#         x = self.res1(x, ax)
#         # assert not torch.isnan(x).any(), "NaN detected after res1"

#         # DyT2 + project + mlp
#         dyt2_out = self.DyT2(x[...,1:])
#         # assert not torch.isnan(dyt2_out).any(), "NaN detected after DyT2"
        
#         proj_dyt2_out = self.project(dyt2_out)
#         # assert not torch.isnan(proj_dyt2_out).any(), "NaN detected after project(DyT2)"
        
#         mlp_out = self.mlp(proj_dyt2_out)
#         # assert not torch.isnan(mlp_out).any(), "NaN detected after mlp"

#         # Residual 2
#         x = self.res2(x, mlp_out)
#         # assert not torch.isnan(x).any(), "NaN detected after res2"
#         # x = self.DyT3(x[...,1:])
#         # x = self.project(x)

#         return x


# class DyT(nn.Module):
#     def __init__(self, d_model, init_alpha = 0.5):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))
#         self.activation = nn.Tanh()
#     def forward(self, x):
#         x = self.activation(self.alpha * x)
#         return self.gamma * x + self.beta


def flash_attn_forward(
    self,
    manifold,
    query_states,
    key_states,
    value_states,
    scale,
    bias,
    mask,
    training=False,
):
    flash_attn_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    query_states.narrow(-1, 0, 1).mul_(-1)

    attn_output, _ = flash_attn_interface(
        self,
        query_states,
        key_states,
        value_states,
        causal=False,
        dropout_p=0.0,
        training=training,
        attention_mask=mask,
    )
    ave = attn_output
    denom = (-manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-6).sqrt()
    att_output = manifold.c.sqrt() * ave / denom
    return att_output


def flex_attn_forward(
    self,
    manifold,
    query_states,
    key_states,
    value_states,
    scale,
    bias,
    mask: BlockMask = None,
    training=False,
):
    # no learnable scaling and bias, remove detach() to enable backprop, but it causes an overhead with torch 2.6.0
    # maybe it is fixed in later versions (2.7.0+)
    # bias is not important as it does not change softmax result, added here for consistency with hnn implementation
    # learnable scale may not be necessary as well
    # maybe using vanilla flash attention would work just fine
    # _scale = (scale / 2).to(torch.float32).detach()
    # _bias = (bias + manifold.c / scale).to(torch.float32).detach()

    query_states.narrow(-1, 0, 1).mul_(-1)
    N_Q_HEADS = query_states.size(1)
    N_KV_HEADS = key_states.size(1)

    # def score_mod(score, b, h, q_idx, kv_idx):
    #     return score * _scale[h] + _bias[h]

    attn_output = flex_attention_compiled(
        query_states,
        key_states,
        value_states,
        block_mask=mask,
        # scale=1.0,
        # score_mod=score_mod,
        enable_gqa=(N_Q_HEADS != N_KV_HEADS),
    )

    ave = attn_output
    denom = (-manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-6).sqrt()
    att_output = manifold.c.sqrt() * ave / denom
    att_output = att_output.transpose(1, 2) # [B, N, H, D]

    return att_output


def eager_attn_forward(module, manifold, query_states, key_states, value_states,
                       scale=None, bias=None, mask=None, training=True, **kwargs):
    att_weight = 2 * manifold.c + 2 * manifold.cinner(query_states, key_states)  # [B, H, N, N]
    # att_weight = manifold.cinner(query_states, key_states)  # [B, H, N, N]
    _scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, H, 1, 1]
    # bias does not change softmax result, added here for consistency with hnn implementation
    _bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    att_weight = att_weight / _scale + _bias  # [B, H, N, N]
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = (mask == 0)

        if mask.dim() == 3 and mask.shape[-2] == 1:
            mask = mask.unsqueeze(-3)  # [B, 1, 1, K]

        # If it's [B, K], expand to [B, 1, 1, K]
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)

        att_weight = att_weight.masked_fill(mask, float("-inf"))

    att_weight = nn.Softmax(dim=-1)(att_weight)  # [B, H, N, N]
    att_output = manifold.lorentzian_centroid(value_states, att_weight)  # [B, H, N, D]
    att_output = att_output.transpose(1, 2) # [B, N, H, D]

    return att_output


ATTN_INTERFACES = {
    'flex_attention': flex_attn_forward,
    'eager': eager_attn_forward,
    'flash_attention_2': flash_attn_forward,
}

class _LorentzMultiheadAttention(nn.Module):
    def __init__(self, manifold: Lorentz,  in_channels, out_channels, num_heads, use_weight=True, power_k=2.0, attn_implementation='flex_attention', trans_heads_concat=False, normalize=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.Wk = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))
        self.Wq = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))
        self.Wv = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(num_heads * out_channels)] * num_heads))
        self.bias = nn.Parameter(torch.zeros(num_heads))

        self.power_k = power_k
        self.trans_heads_concat = trans_heads_concat
        if self.trans_heads_concat:
            self.final_linear = nn.Linear(self.num_heads * (self.out_channels), self.num_heads * self.out_channels - 1) #should be nn.linear instead of LorentLinear
        self.normalize = normalize
        self.attn_implementation = attn_implementation
        self.is_causal = False

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def apply_rotary_embeddings(self, x, freqs_complex, device):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attentions=False, mask=None, rot_pos=None):
        batch_size, seq_length, embed_dim = source_input.size()

        query = self.Wq(source_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]
        key = self.Wk(source_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]
        value = self.Wv(source_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]

        if rot_pos is not None:
            query = self.apply_rotary_embeddings(query, rot_pos, query.device)
            key = self.apply_rotary_embeddings(key, rot_pos, key.device)


        # reshape the inputs
        query_states = self.project(query).transpose(1, 2)
        key_states = self.project(key).transpose(1, 2)
        value_states = self.project(value).transpose(1, 2)
        # # normalize input
        if self.normalize:
            query_states = LorentzNormalization(self.manifold)(query_states)
            key_states = LorentzNormalization(self.manifold)(key_states)

        attn_interface = ATTN_INTERFACES[self.attn_implementation]

# AFTER (positional args only; order matters!)
        att_output = attn_interface(
            self,
            self.manifold,
            query_states,
            key_states,
            value_states,
            self.scale,
            self.bias,
            mask,
            self.training,
        )

        if self.trans_heads_concat:
            att_output_space = self.final_linear(att_output.reshape(att_output.size(0), att_output.size(1), self.num_heads * self.out_channels))
            att_output_time = ((att_output_space**2).sum(dim=-1, keepdims=True) + self.manifold.c).sqrt()
            att_output = torch.cat([att_output_time, att_output_space], dim=-1)     
            att_output = att_output       
        else:
            att_output = self.manifold.lorentzian_centroid(att_output)

        return att_output


class _LTransformerBlock(nn.Module):
    """
    A single Transformer block for the decoder.
    - Uses **masked** self-attention (causal).
    - Uses hyperbolic normalization and activation.
    """

    def __init__(self, manifold, d_model: int, n_head: int, attn_implementation='flex_attention'):
        super().__init__()
        dim_per_head = d_model // n_head
        self.manifold = manifold

        self.attn = _LorentzMultiheadAttention(
            manifold, dim_per_head, dim_per_head, n_head,
            attn_implementation=attn_implementation, trans_heads_concat=True
        )

        self.ln_1 = hnn.LorentzRMSNorm(manifold, d_model - 1)

        # MLP (Feed-forward network)
        self.mlp = LorentzFeedForward(manifold, d_model, d_model * 4)


        self.ln_2 = hnn.LorentzRMSNorm(manifold, d_model - 1)
        self.ln_3 = hnn.LorentzRMSNorm(manifold, d_model - 1)
        self.res1 = hnn.LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))
        self.res2 = hnn.LResNet(manifold, use_scale=True, scale=4.0 * math.sqrt(d_model))

    def forward(self, x, attn_mask=None, rope=None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask, rot_pos=rope)  # Masked attention
        x = self.res1(x, ax)
        x = self.ln_3(self.res2(x, self.mlp(self.ln_2(x))))
        return x

class LTransformerEncoder(nn.Module):
    """
    Text encoder using multiple layers of transformer encoder blocks. It accepts
    tokenized text sequences, passes them through word/position embedding layers
    and further processes them through transformer layers.
    """

    def __init__(
        self,
        manifold_in: Lorentz, 
        manifold_hidden: Lorentz,
        manifold_out: Lorentz,
        arch: str,
        vocab_size: int,
        context_length: int,
        grad_checkpointing: bool = False,
        attn_implementation: str = 'flex_attention',
    ):
        """
        Args:
            arch: Architecture config for transformer, describing layers, width,
                and number of attention heads. For example, `L12_W512_A8` has 1
                layer, 512 width, 8 heads. Width of MLP will always be `4 * W`,
                per transformer paper. `A` is optional and will default to
                (`A = H/64`) per transformer paper.
            vocab_size: Number of tokens in the output vocabulary.
            context_length: Maximum length of input captions; this is used to
                create a fixed positional embedding lookup table.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        # Parse architecture str: layers, width, heads, feed-forward size.
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))

        # Find heads in architecture else use (H // 64) per (Vaswani et al.)
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64

        # Input sequences in forward pass will be right padded with zeroes.
        # `nn.Embedding` has a `padding_idx` argument to set their embedding as
        # zero. However, since the blocks are uni-directional, they will never
        # receive gradients for padded positions.
        self.token_embed = hnn.LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden) #this step automatically adds the positional embedding

        # Make a sequential module of transformer encoder blocks.
        _resblocks = [
            _LTransformerBlock(manifold_hidden, self.width, self.heads, attn_implementation=attn_implementation) for _ in range(self.layers)
        ]
        self.resblocks = nn.ModuleList(_resblocks)
        self.ln_final = hnn.LorentzLayerNorm(manifold_out, self.width - 1)
        self.final_proj = hnn.LorentzLinear(manifold_hidden, self.width, self.width - 1, manifold_out=manifold_out)

        # Generate a unidirectional mask for self-attention. As per PyTorch API,
        # masked positions are set to `-inf`.
        # attn_mask = None # torch.triu(
        #     torch.full((context_length, context_length), float("-inf")), diagonal=1
        # )
        # self.register_buffer("attn_mask", attn_mask.bool())

        # Initialize all modules like CLIP:
        # nn.init.normal_(self.token_embed.weight, std=0.02)
        # nn.init.normal_(self.posit_embed.data, std=0.01)

        out_proj_std = (2 * self.width * self.layers) ** -0.5
        # for block in self.resblocks:
        #     nn.init.normal_(block.mlp[0].linear.weight, std=(self.width) ** -0.5)
        #     nn.init.normal_(block.mlp[2].linear.weight, std=out_proj_std)

    def set_attn_implementation(self, attn_implementation: str):
        for block in self.resblocks:
            block.attn.attn_implementation = attn_implementation

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Obtain features of input text tokens by passing them through transformer
        blocks. All self-attention layers only attend to past token (left side).
        """

        max_len = text_tokens.shape[-1]
        # _attn_mask = self.attn_mask[:max_len, :max_len]

        # shape: (batch_size, context_length, width)
        token_embeddings = self.token_embed(text_tokens)

        # Forward pass through transformer, optionally with grad checkpointing.
        textual_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                # shape: (context_length, batch_size, width)
                textual_features = checkpoint(block, textual_features, None)
            else:
                textual_features = block(textual_features, None)
        textual_features = self.final_proj(textual_features)
        textual_features = self.ln_final(textual_features)
        return textual_features
