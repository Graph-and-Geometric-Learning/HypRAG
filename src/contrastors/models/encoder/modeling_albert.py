import logging
import os
import math
from functools import partial
from typing import List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from einops import rearrange
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertForPreTrainingOutput,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertAttention,
    ModernBertMLP,
)

from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN
from dataclasses import dataclass

from layers.embedding import BertEmbeddings
import hyplib.nn as hnn
from hyplib.manifolds import Lorentz
from hyplib.models import LorentzFeedForward
from hyplib.nn.conv import LorentzNormalization, LorentzActivation 
from hyplib.models.Transformer_encoder import _LTransformerBlock

from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .configuration_albert import ALBertConfig

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


logger = logging.getLogger(__name__)

@dataclass
class ALBertOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    manifold: Optional[Lorentz] = None
    info: Optional[dict] = None


@dataclass
class ALBertForPreTrainingOutput(BertForPreTrainingOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    info: Optional[dict] = None


class ALBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = ALBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    loss_type = "ForMaskedLM"
    _no_split_modules = ["_LTransformerBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        if not isinstance(config, ALBertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `ALBertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ALBertModel):
            module.gradient_checkpointing = value


def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class CLSPooler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, input_ids=None, attention_mask=None):
        return hidden_states[:, 0]


class EuclideanMeanPooler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is None:
            return torch.mean(hidden_states, dim=1)

        s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(axis=1, keepdim=True).float()
        return s / d


class OutwardEinsteinMidpointPooler(nn.Module):
    def __init__(self, config, manifold, p: float = 1.0, beta: float = 0.0, eps: float = 1e-12):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.p = p
        self.beta = beta
        self.eps = eps

    def forward(self, hidden_states, attention_mask=None):
        x0 = hidden_states[..., :1]
        xsp = hidden_states[..., 1:]

        k = xsp / (x0 + self.eps)

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            w = m
            denom_w = w.sum(dim=1, keepdim=True).clamp(min=self.eps)
            w = w / denom_w
        else:
            B, S, _ = x0.shape
            w = torch.ones_like(x0) / float(S)
        k_norm2 = (k * k).sum(dim=-1, keepdim=True)
        one_minus = (1.0 - k_norm2).clamp(min=self.eps)
        gamma = one_minus.rsqrt()

        phi = gamma.pow(self.p)
        if self.beta > 0.0:
            phi = phi / (1.0 + self.beta * phi)

        wtil = w * phi
        num = (wtil * gamma) * k
        den = (wtil * gamma).sum(dim=1, keepdim=True).clamp(min=self.eps)
        m_k = num.sum(dim=1, keepdim=True) / den

        ret = self.project(m_k.squeeze(1))
        return ret

    def project(self, k):
        c = getattr(self.manifold, "c", 1.0)
        sqrt_c = torch.sqrt(torch.tensor(c, dtype=k.dtype, device=k.device))

        k_norm2 = (k * k).sum(dim=-1, keepdim=True)
        one_minus = (1.0 - k_norm2).clamp(min=self.eps)
        denom = torch.sqrt(one_minus)

        x0 = sqrt_c / denom
        xi = (sqrt_c * k) / denom
        return torch.cat([x0, xi], dim=-1)


class AttentiveMeanPooler(nn.Module):
    def __init__(self, config, manifold):
        super().__init__()
        self.config = config
        self.manifold = manifold

        self.q_proj = hnn.LorentzLinear(manifold, config.hidden_size, config.hidden_size - 1)
        self.kv_proj = hnn.LorentzLinear(manifold, config.hidden_size, config.hidden_size - 1)

    def forward(self, hidden_states, attention_mask=None):
        cls_token = hidden_states[:, :1, :]
        # rest_tokens = hidden_states[:, 1:, :]
        query = self.q_proj(cls_token)
        # key = self.kv_proj(rest_tokens)
        # value = self.kv_proj(rest_tokens)
        key = self.kv_proj(hidden_states)
        value = self.kv_proj(hidden_states)

        weights = self.manifold.pairwise_inner(query, key)  # (B, 1, S)
        if attention_mask is not None:
            # mask = attention_mask[:, 1:].unsqueeze(1).bool()  # (B, 1, S - 1)
            mask = attention_mask.unsqueeze(1).bool()  # (B, 1, S)
            weights = weights.masked_fill(~mask, float("-inf"))
        attn_weight = nn.Softmax(dim=-1)(weights)  # [B, H, N, N]
        attn_output = self.manifold.lorentzian_centroid(value, attn_weight)  # (B, D, 1)
        return attn_output.squeeze(1)


POOLER_LAYERS = {
    "cls": CLSPooler,
    "outward_einstein_midpoint": OutwardEinsteinMidpointPooler,
    "attentive_mean": AttentiveMeanPooler,
}


class LorentzDyTNorm(nn.Module):
    def __init__(self, manifold, hidden_size, init_alpha=0.5):
        super().__init__()
        self.manifold = manifold
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.activation = nn.Tanh()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.activation(self.alpha * x_space)
        x_space = self.gamma * x_space + self.beta
        x = self.manifold.add_time(x_space)
        return x


NORM_LAYERS = {
    "layer_norm": hnn.LorentzLayerNorm,
    "rms_norm": hnn.LorentzRMSNorm,
    "dyt_norm": LorentzDyTNorm,
}


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class ALBertRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.attention_scaling = 1.0
        self.head_space_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) - 1
        self.rope_dim = self.head_space_dim - (self.head_space_dim % 2)  # must be even

        self.config = config

        inv_freq = self.rope_init(
            self.max_seq_len_cached,
            self.rope_dim,
            base=config.rope_theta,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def rope_init(self, max_seq_len, dim, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype), self.rope_dim


def apply_rotary_pos_emb(x, cos, sin, rope_dim, unsqueeze_dim=1):
    """
    x: [batch, heads, seq, dim]
    cos/sin: [seq, rope_dim] (will be broadcast across batch/heads)
    rope_dim <= dim and is even.
    Last (dim - rope_dim) channels are passed through unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]  # possibly 1 channel if original dim was odd

    # Rotate in 2D pairs: [x0, x1, x2, x3, ...] -> pairwise rotation
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]

    x_rot_out_even = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
    x_rot_out_odd  = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]

    x_rot_out = torch.empty_like(x_rot)
    x_rot_out[..., 0::2] = x_rot_out_even
    x_rot_out[..., 1::2] = x_rot_out_odd

    if x_pass.numel() == 0:
        return x_rot_out
    return torch.cat([x_rot_out, x_pass], dim=-1)


class ALBertEmbeddings(nn.Module):
    def __init__(self, config, manifold: Lorentz = None):
        super().__init__()
        self.config = config
        self.manifold = manifold or Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size - 1, padding_idx=config.pad_token_id)

        self.pad_token_id = config.pad_token_id
        self.add_abs_pos_emb = (config.use_rotary_embeddings == False)
        if self.add_abs_pos_emb:
            assert config.max_position_embeddings > 0, "if using absolute position embeddings, max_position_embeddings should be > 0"
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size - 1,
                padding_idx=config.pad_token_id,
            )

        self.type_vocab_size = getattr(config, "type_vocab_size", 0)
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size - 1)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        seqlen = input_ids.size(1)
        embeddings = self.word_embeddings(input_ids)
        if self.add_abs_pos_emb:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        # Add time-like dimension
        embeddings = self.manifold.add_time(embeddings)

        return embeddings


def flash_attn_forward(
    module,
    manifold,
    query_states,
    key_states,
    value_states,
    scale,
    attention_mask,
    training=False,
):
    flash_attn_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    query_states.narrow(-1, 0, 1).mul_(-1)
    # Ugly workaround for flash attention dtype issue during inference
    if query_states.dtype not in [torch.float16, torch.bfloat16] or \
         key_states.dtype not in [torch.float16, torch.bfloat16] or \
            value_states.dtype not in [torch.float16, torch.bfloat16]:
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)

    attn_output, _ = flash_attn_interface(
        module,
        query_states,
        key_states,
        value_states,
        causal=False,
        dropout_p=0.0,
        training=training,
        attention_mask=attention_mask,
    )
    ave = attn_output
    denom = (-manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-6).sqrt()
    att_output = manifold.c.sqrt() * ave / denom
    return att_output


def flex_attn_forward(
    module,
    manifold,
    query_states,
    key_states,
    value_states,
    scale,
    attention_mask: BlockMask = None,
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

    # Check if head dimension is not a power of 2, do padding
    head_dim = query_states.size(-1)
    if (head_dim & (head_dim - 1)) != 0:
        next_pow2 = 1 << (head_dim - 1).bit_length()
        pad_size = next_pow2 - head_dim
        query_states = F.pad(query_states, (0, pad_size))
        key_states = F.pad(key_states, (0, pad_size))
        value_states = F.pad(value_states, (0, pad_size))
    else:
        pad_size = 0

    # def score_mod(score, b, h, q_idx, kv_idx):
    #     return score * _scale[h] + _bias[h]

    attn_output = flex_attention_compiled(
        query_states,
        key_states,
        value_states,
        block_mask=attention_mask,
        # scale=1.0,
        # score_mod=score_mod,
        enable_gqa=(N_Q_HEADS != N_KV_HEADS),
    )

    ave = attn_output
    denom = (-manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-6).sqrt()
    att_output = manifold.c.sqrt() * ave / denom

    if pad_size > 0:
        att_output = att_output[..., :head_dim]

    att_output = att_output.transpose(1, 2) # [B, N, H, D]
    return att_output


def eager_attn_forward(module, manifold, query_states, key_states, value_states,
                       scale=None, mask=None, training=True, **kwargs):
    att_weight = 2 * manifold.c + 2 * manifold.cinner(query_states, key_states)  # [B, H, N, N]
    # att_weight = manifold.cinner(query_states, key_states)  # [B, H, N, N]
    _scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, H, 1, 1]
    att_weight = att_weight / _scale
    if mask is not None:
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


class LorentzLinear(nn.Module):
    def __init__(self, manifold: Lorentz, in_features, out_features, bias=True, eps=1e-8):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.bias = bias

        self.linear = nn.Linear(self.in_features, self.out_features-1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x_space = self.linear(x)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.c).clamp_min(self.eps).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return x


class LorentzMultiHeadLinear(nn.Module):
    def __init__(self, manifold: Lorentz, in_features, out_features, num_heads, bias=True, eps=1e-8):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.eps = eps
        self.bias = bias

        self.linear = nn.Linear(
            self.in_features,
            (self.out_features-1) * self.num_heads,
            bias=bias
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, position_embeddings=None):
        x_space = self.linear(x).view(*x.shape[:-1], self.num_heads, self.out_features - 1)
        if position_embeddings is not None:
            cos, sin, rope_dim = position_embeddings
            x_space = apply_rotary_pos_emb(x_space, cos, sin, rope_dim, unsqueeze_dim=-2)

        x = self.manifold.add_time(x_space)
        return x


class ALBertMultiheadAttention(nn.Module):
    def __init__(self, config, layer_idx:int, manifold: Lorentz = None):
        super().__init__()
        self.config = config
        self.manifold = manifold or Lorentz(config.curvature_init, learnable=config.trainable_curvature)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False
        self.query_key_norm = getattr(config, "query_key_norm", False)
        self.attn_heads_concat = getattr(config, "attn_heads_concat", True)
        self.attn_implementation = config.attn_implementation
        self.layer_idx = layer_idx

        self.q_proj = LorentzMultiHeadLinear(
            self.manifold, config.hidden_size, self.head_dim, config.num_attention_heads, bias=config.attention_bias
        )
        self.k_proj = LorentzMultiHeadLinear(
            self.manifold, config.hidden_size, self.head_dim, config.num_attention_heads, bias=config.attention_bias
        )
        self.v_proj = LorentzMultiHeadLinear(
            self.manifold, config.hidden_size, self.head_dim, config.num_attention_heads, bias=config.attention_bias
        )

        if self.query_key_norm:
            self.q_norm = LorentzNormalization(self.manifold)
            self.k_norm = LorentzNormalization(self.manifold)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if self.attn_heads_concat:
            self.concat_proj = nn.Linear(self.head_dim * config.num_attention_heads, config.hidden_size - 1)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_norm(self.q_proj(hidden_states, position_embeddings)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states, position_embeddings)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).transpose(1, 2)
        attn_interface = ATTN_INTERFACES["flash_attention_2"]

        attn_output = attn_interface(
            self,
            self.manifold,
            query_states,
            key_states,
            value_states,
            self.scaling,
            attention_mask,
            self.training,
        )

        if self.attn_heads_concat:
            attn_output_space = self.concat_proj(attn_output.reshape(*input_shape, -1))
            attn_output = self.manifold.add_time(attn_output_space)
        else:
            attn_output = self.manifold.lorentzian_centroid(attn_output)

        return attn_output


class ALBertEncoderLayer(nn.Module):
    def __init__(self, config, layer_idx:int, manifold=None):
        super().__init__()
        self.config = config
        self.manifold = manifold or Lorentz(config.curvature_init, learnable=config.trainable_curvature)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.norm_layer = NORM_LAYERS.get(config.norm_layer, 'rms_norm')

        self.self_attn = ALBertMultiheadAttention(config, layer_idx, manifold=self.manifold)

        norm_layer_cls = NORM_LAYERS.get(config.norm_layer, 'rms_norm')
        self.input_norm = norm_layer_cls(self.manifold, config.hidden_size - 1)

        # MLP (Feed-forward network)
        self.mlp = LorentzFeedForward(self.manifold, config.hidden_size, config.hidden_size * 4)

        self.post_attention_norm = self.norm_layer(self.manifold, config.hidden_size - 1)

        self.residual_add = hnn.LResNet(self.manifold, use_scale=True, scale=4.0 * math.sqrt(config.hidden_size))
        self.post_mlp_residual_add = hnn.LResNet(self.manifold, use_scale=True, scale=4.0 * math.sqrt(config.hidden_size))

        self.output_norm = self.norm_layer(self.manifold, config.hidden_size - 1)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )

        hidden_states = self.residual_add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_residual_add(residual, hidden_states)

        hidden_states = self.output_norm(hidden_states)
        return hidden_states


class ALBertModel(ALBertPreTrainedModel):

    def __init__(self, config: ALBertConfig, manifold: Lorentz = None, add_pooling_layer=True):
        super().__init__(config)

        self.manifold = manifold or Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self.embeddings = ALBertEmbeddings(config, manifold=self.manifold)

        self.layers = nn.ModuleList([
            ALBertEncoderLayer(config, self.manifold) for _ in range(config.num_hidden_layers)
            # _LTransformerBlock(self.manifold, config.hidden_size, config.num_attention_heads, attn_implementation=config.attn_implementation) for _ in range(config.num_hidden_layers)
        ])

        self.final_proj = LorentzLinear(self.manifold, config.hidden_size, config.hidden_size)
        self.norm_layer_cls = NORM_LAYERS.get(config.norm_layer, 'rms_norm')
        self.final_norm = self.norm_layer_cls(self.manifold, config.hidden_size - 1)

        if self.config.use_rotary_embeddings:
            self.rotary_emb = ALBertRotaryEmbedding(config)

        self.pooling = config.pooling
        if self.pooling in POOLER_LAYERS:
            pooler_cls = POOLER_LAYERS.get(self.pooling)
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")
        self.pooler = pooler_cls(config, self.manifold) if add_pooling_layer else None

        self.grad_checkpointing = getattr(config, "gradient_checkpointing", False)

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        if position_ids is None:
            if self.config.pad_token_id is not None and self.config.pad_token_id > 0:
                position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id).to(input_ids.device)
            else:
                seqlen = input_ids.size(1)
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bsz, seq_len)

        hidden_states = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        attention_mask = self.prepare_attention_mask(input_ids, attention_mask)
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.config.use_rotary_embeddings else None

        # Forward pass through transformer, optionally with grad checkpointing.
        for i, block in enumerate(self.layers):
            if self.grad_checkpointing and self.training:
                # shape: (context_length, batch_size, width)
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                )

        hidden_states = self.final_proj(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        info = {}
        with torch.no_grad():
            hyp_time = hidden_states[..., 0].detach()
            info["hyperbolic_time"] = hyp_time.mean()
            info["hyperbolic_time_std"] = hyp_time.std()
            info["curvature"] = self.manifold.k.detach()

        return ALBertOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
            manifold=self.manifold,
            info=info,
        )

    def prepare_attention_mask(self, input_ids, attention_mask):
        if self.config.attn_implementation == 'eager':
            def build_mask(mask, batch_size: int, num_heads: int, seq_len: int):
                if mask is None:
                    return None
                if mask.dim() == 4:
                    if mask.shape == (batch_size, num_heads, seq_len, seq_len):
                        return mask
                    else:
                        raise ValueError(f"Mask with 4 dims must be [B, H, N, N]")

                if mask.dim() == 2:
                    m, n = mask.shape
                    if (m == seq_len) and (n == seq_len):
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    else: # [B, N]
                        mask = mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, N]
                    return mask

                if mask.dim() == 3:
                    b, m, n = mask.shape
                    # [B, N, N]
                    if (m == seq_len) and (n == seq_len):
                        mask = mask.unsqueeze(1)  # [B, 1, N, N]
                        return mask

                    # [B, 1, N]
                    if (m == 1) and (n == seq_len):
                        mask = mask.squeeze(1)
                        mask = mask.unsqueeze(1).unsqueeze(2)
                        return mask

                    # [B, H, N]
                    if (m == num_heads) and (n == seq_len):
                        mask = mask.unsqueeze(2)
                        return mask
            mask = build_mask(attention_mask, input_ids.shape[0], self.config.n_head, input_ids.shape[1]) if attention_mask is not None else None
            return mask



class ALBertPredictionHeads(nn.Module):
    def __init__(self, config, manifold=None):
        super().__init__()
        self.config = config
        self.manifold = manifold or Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self.dense = LorentzLinear(self.manifold, config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.act_fn = ACT2FN[config.classifier_activation]
        self.act = LorentzActivation(self.manifold, self.act_fn)
        self.norm_layer_cls = NORM_LAYERS.get(config.norm_layer, 'rms_norm')
        self.norm = self.norm_layer_cls(self.manifold, config.hidden_size - 1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LorentzMLR(nn.Module):
    def __init__(
            self,
            manifold: Lorentz,
            num_features: int,
            num_classes: int,
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes, num_features-2), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1/self.manifold.k.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK*self.a)*norm_z)
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(-w_t**2+torch.norm(w_s, dim=-1)**2)
        alpha = -w_t*x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK*self.a)*torch.inner(x.narrow(-1, 1, x.shape[-1]-1), self.z))

        d = self.manifold.k.sqrt()*torch.abs(torch.asinh(sqrt_mK*alpha/beta))  # Distance to hyperplane
        logits = torch.sign(alpha)*beta*d

        return logits
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class ALBertForPreTraining(ALBertPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config: ALBertConfig):
        super().__init__(config)
        self.config = config

        self.manifold = Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self.model = ALBertModel(
            config,
            manifold=self.manifold,
            add_pooling_layer=getattr(config, "add_pooling_layer", False)
        )

        self.head = ALBertPredictionHeads(config, manifold=self.manifold)
        # self.head = nn.Identity()
        # self.decoder = nn.Linear(
        #     config.hidden_size - 1, config.vocab_size, bias=config.decoder_bias
        # )
        self.decoder = LorentzMLR(
            self.manifold,
            config.hidden_size,
            config.vocab_size,
        )

        # TODO: maybe need to implement a hyperbolic equivalent of this
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

        if config.tie_word_embeddings:
            self.tie_weights()

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index
        self.loss_type = "ForMaskedLM"

    def tie_weights(self):
        """ Tie weights between the input embeddings and the output embeddings. """
        self._tied_weights_keys.append("model.embeddings.word_embeddings.weight")
        self._tied_weights_keys.append("decoder.weight")

        self.decoder.weight = self.model.embeddings.word_embeddings.weight

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
        )

        last_hidden_state = outputs[0]
        # last_hidden_state = last_hidden_state[:, :, 1:]  # remove time-like dimension for prediction head

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ALBertForPreTrainingOutput(
            loss=loss,
            logits=logits,
            info=outputs.info,
        )
