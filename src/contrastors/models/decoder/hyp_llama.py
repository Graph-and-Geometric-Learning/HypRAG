from datasets import load_from_disk
import torch
from collections import OrderedDict
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import hyplib.nn as hnn
from hyplib.manifolds import Lorentz
import re
from hyplib.optimizers import Optimizer, LR_Scheduler
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Literal
import numpy as np
from dataclasses import dataclass
import os
import torch.nn.functional as F
import wandb
from hyplib.models import LorentzFeedForward
from transformers import AutoTokenizer
import random

def precompute_theta_pos_frequencies(head_dim, seq_len, theta: float = 10000.0):
    head_dim -= 1
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)) # (Head_Dim / 2)
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class _LTransformerDecoderBlock(nn.Module):
    """
    A single Transformer block for the decoder.
    - Uses **masked** self-attention (causal).
    - Uses hyperbolic normalization and activation.
    """

    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        self.manifold = manifold

        # Masked self-attention (only looks at past tokens)
        self.attn = hnn.LorentzMultiheadAttention(
            manifold, dim_per_head, dim_per_head, n_head,
            attention_type='full', trans_heads_concat=True
        )

        self.ln_1 = hnn.LorentzRMSNorm(manifold, d_model - 1)

        # MLP (Feed-forward network)
        self.mlp = LorentzFeedForward(manifold, d_model, d_model * 4)

        self.ln_2 = hnn.LorentzRMSNorm(manifold, d_model - 1)
        self.res1 = hnn.LResNet(manifold, use_scale=True, scale=(2 * math.sqrt(d_model)))
        self.res2 = hnn.LResNet(manifold, use_scale=True, scale=(2 * math.sqrt(d_model)))

    def forward(self, x, attn_mask=None, rope=None):
        """
        Forward pass with causal attention mask.
        """
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, output_attentions=False, mask=attn_mask, rot_pos=rope)  # Masked attention
        x = self.res1(x, ax)
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x


class LTransformerDecoder(nn.Module):
    """
    A decoder-only Transformer (like LLAMA) that:
    - Uses **causal attention mask** (future tokens are masked).
    - Outputs **logits** for next-token prediction.
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        # Parse architecture string
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64
        # Token Embeddings (Lorentz)
        self.token_embed = hnn.LorentzEmbeddings(manifold_in, vocab_size, self.width, manifold_out=manifold_hidden, posit_embed=False)  # Adds positional embedding automatically

        # Transformer Blocks (Decoder Only)
        self.resblocks = nn.ModuleList([
            _LTransformerDecoderBlock(manifold_hidden, self.width, self.heads)
            for _ in range(self.layers)
        ])

        # Final normalization and projection
        self.ln_final = hnn.LorentzRMSNorm(manifold_hidden, self.width - 1)
        self.final_proj = hnn.LorentzLinear(
            manifold_hidden, self.width, self.width - 1, manifold_out=manifold_hidden
        )
        self.mapping = hnn.LorentzLinear(
            manifold_hidden, self.width, self.vocab_size, manifold_out=manifold_out
        )

        # **Causal Attention Mask (Precomputed)**
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())
        rope_vals = precompute_theta_pos_frequencies(self.width// self.heads, self.context_length)
        self.register_buffer("freqs_complex", rope_vals)
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Given input tokens, return logits for next-token prediction.
        """
        max_len = text_tokens.shape[-1]
        _attn_mask = self.attn_mask[:max_len, :max_len]  # Apply correct mask

        # shape: (batch_size, context_length, width)
        token_embeddings = self.token_embed(text_tokens)


        # Forward pass through Transformer blocks
        decoder_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                decoder_features = checkpoint(block, decoder_features, _attn_mask, self.freqs_complex)
            else:
                decoder_features = block(decoder_features, _attn_mask, self.freqs_complex)

        decoder_features = self.final_proj(decoder_features)
        decoder_features = self.ln_final(decoder_features)

        # shape: (batch_size, context_length, hidden_dim(width)+1)
        # next, map to vocab size
        logits = self.mapping(decoder_features)[..., 1:]
        # shape: (batch_size, context_length, vocab_size)
        assert(not logits.isnan().any())
        assert(not logits.isinf().any())
        return logits

access_token = '...'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token  

manifold_in = Lorentz(1.0)
manifold_hidden = Lorentz(1.0)
manifold_out = Lorentz(1.0)
# Define model
decoder = LTransformerDecoder(
    manifold_in=manifold_in,
    manifold_hidden=manifold_hidden,
    manifold_out=manifold_out,
    arch="L6_W390_A6",  
    vocab_size=128256,     #vocab size of llama3.1-8B tokenizer
    context_length=2048
)
