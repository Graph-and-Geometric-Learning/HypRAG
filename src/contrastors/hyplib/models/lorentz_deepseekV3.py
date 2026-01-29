import hyplib.nn as hnn
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from hyplib.manifolds import Lorentz
from hyplib.models.lorentz_MoE import LorentzMoE

@dataclass
class DeepSeekModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2049
    inter_dim: int = 10945
    moe_inter_dim: int = 1409
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 513
    qk_nope_head_dim: int = 129
    qk_rope_head_dim: int = 65
    v_head_dim: int = 129
    # yarn
    original_seq_len: int = 4097
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


def precompute_freqs_cis(args: DeepSeekModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim - 1
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


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

class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, manifold: Lorentz, layer_id: int, args: DeepSeekModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (DeepSeekModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.manifold = manifold
        self.attn = hnn.LorentzMLA(self.manifold, args)
        self.ffn = LorentzMLP(manifold, args.dim, args.inter_dim) if layer_id < args.n_dense_layers else LorentzMoE(manifold, args)
        self.attn_norm = hnn.LorentzRMSNorm(self.manifold, args.dim - 1)
        self.ffn_norm = hnn.LorentzRMSNorm(self.manifold, args.dim - 1)
        self.attn_res = hnn.LResNet(self.manifold, use_scale=True, scale=10.0)
        self.ffn_res = hnn.LResNet(self.manifold, use_scale=True, scale=10.0)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = self.attn_res(x, self.attn(self.attn_norm(x), start_pos, freqs_cis, mask))
        x = self.ffn_res(x, self.ffn(self.ffn_norm(x)))
        return x


class LorentzDeepSeekV3(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        manifold_in (Lorentz): input manifold
        manifold_hidden (Lorentz): intermediate embedding manifold
        manifold_out (Lorentz): output manifold
        max_seq_len (int): Maximum sequence length for the transformer.
        embed: Lorentz word embedding layer for input tokens.
        layers: List of Lorentz transformer blocks.
        norm: Lorentz RMS layer normalization applied after all blocks.
        head: Output projection layer mapping to vocabulary size, Lorentz linear layer
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: DeepSeekModelArgs, manifold_in, manifold_hidden, manifold_out):
        """
        Initializes the Transformer model.

        Args:
            args (DeepSeekModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()

        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.max_seq_len = args.max_seq_len
        self.embed = hnn.LorentzEmbeddings(self.manifold_in, args.vocab_size, args.dim, self.manifold_hidden)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(manifold_hidden, layer_id, args))
        self.norm = hnn.LorentzRMSNorm(self.manifold_hidden, args.dim - 1)

        self.head = hnn.LorentzLinear(self.manifold_hidden, args.dim, args.vocab_size - 1, self.manifold_in)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def project(self, manifold, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits[..., 1:]) for _ in range(world_size)]
            dist.all_gather(all_logits, logits[..., 1:])
            logits = self.project(self.manifold_out, torch.cat(all_logits, dim=-1))
        return logits