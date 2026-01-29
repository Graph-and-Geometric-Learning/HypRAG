# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import os
import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.ops.rms_norm import RMSNorm, rms_norm
from safetensors.torch import load_file as safe_load_file
from transformers import GPT2Config, PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import (
    BertForPreTrainingOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_outputs import ModelOutput
from hyplib.models.Transformer_encoder import _LTransformerBlock
from dataclasses import dataclass

import hyplib.nn as hnn
from hyplib.manifolds import Lorentz
from layers import Block
from layers.embedding import BertEmbeddings
from models.encoder.elbert import get_base_hidden_size
from models.encoder.configuration_elbert import ELBertConfig
from models.model_utils import filter_shapes, state_dict_from_pretrained
from megablocks.layers.arguments import Arguments
from megablocks.layers import moe

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.layer_norm import layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None


logger = logging.getLogger(__name__)


@dataclass
class ELBertOutput(ModelOutput):
    euc_hidden_state: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    manifold: Optional[Lorentz] = None
    info: Optional[dict] = None


@dataclass
class ELBertForPreTrainingOutput(BertForPreTrainingOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    info: Optional[dict] = None


class ELBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = ELBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        self.config = config

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ELBertEncoder):
            module.gradient_checkpointing = value


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


ACTIVATION_FNS = {
    "gelu": F.gelu,
    "swiglu": F.silu,
    "glu": F.glu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": F.mish,
}


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
    "euc_cls": CLSPooler,
    "euc_mean": EuclideanMeanPooler,
    "hyp_cls": CLSPooler,
    "hyp_outward_einstein_midpoint": OutwardEinsteinMidpointPooler,
    "hyp_attentive_mean": AttentiveMeanPooler,
}


class DeLayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.learnable_norm_scaler = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        norm_scaler = self.learnable_norm_scaler(hidden_states)
        # norm_scaler = torch.clamp(norm_scaler, min=0.0, max=1.0)
        hidden_states = hidden_states * (norm_scaler + 1.0)
        return hidden_states


class ELBertPredictionHeadTransform(nn.Module):
    def __init__(self, config, hidden_size, manifold=None):
        super().__init__()
        self.manifold = manifold if manifold is not None else Lorentz(config.manifold_out, learnable=config.trainable_curvature)

        self.dense = hnn.LorentzLinear(self.manifold, hidden_size, hidden_size, bias=True)

        if config.activation_function == "swiglu":
            self.act_fn = F.silu
        else:
            approximate = "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
            self.act_fn = nn.GELU(approximate=approximate)
        self.lorentz_act_fn = hnn.LorentzActivation(self.manifold, self.act_fn)

        self.layer_norm = hnn.LorentzRMSNorm(self.manifold, hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.lorentz_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class ELBertLMPredictionHead(nn.Module):
    def __init__(self, config, hidden_size, manifold=None):
        super().__init__()
        self.manifold = manifold if manifold is not None else Lorentz(config.manifold_out)

        self.transform = ELBertPredictionHeadTransform(config, hidden_size, manifold=self.manifold)

        self.decoder = hnn.LorentzMLR(
            manifold, hidden_size, config.base_model_config.vocab_size
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class ELBertModel(ELBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, manifold=None, base_pretrained=True, add_pooling_layer=True):
        super().__init__(config)
        self.manifold = manifold if manifold is not None else Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self._curv_minmax = {
            "max": config.curvature_max if config.curvature_max is not None else math.inf,
            "min": config.curvature_min if config.curvature_min is not None else -math.inf,
        }
        if base_pretrained:
            self.base_encoder = AutoModel.from_pretrained(config.base_model_name, config=config.base_model_config, torch_dtype=torch.bfloat16)
        else:
            if config.base_model_config is None:
                config.base_model_config = AutoConfig.from_pretrained(config.base_model_name)

            self.base_encoder = AutoModel.from_config(config.base_model_config)
        self.base_hidden_size = get_base_hidden_size(config.base_model_config)
        self.hidden_size = config.hidden_size if config.hidden_size else self.base_hidden_size
        config.hidden_size = self.hidden_size  # for compatibility with pooler and other modules

        self.delayer_norm = DeLayerNorm(self.base_hidden_size) if config.delayer_norm else None

        if config.projection == 'linear':
            self.projection = nn.Linear(self.base_hidden_size, self.hidden_size - 1)
        elif config.projection == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(self.base_hidden_size, self.hidden_size - 1),
                nn.GELU(),
                nn.Linear(self.hidden_size - 1, self.hidden_size - 1),
                nn.GELU(),
                nn.Linear(self.hidden_size - 1, self.hidden_size - 1),
            )
            self.projection.apply(partial(_init_weights, initializer_range=config.initializer_range))
        else:
            raise ValueError(f"Projection type {config.projection} not supported")

        self.num_hyperbolic_blocks = getattr(config, "num_hyperbolic_blocks", 0)

        if self.num_hyperbolic_blocks > 0:
            self.n_head = getattr(config.base_model_config, "num_attention_heads", 3)
            self.hyperbolic_blocks = nn.ModuleList([
                _LTransformerBlock(self.manifold, self.hidden_size, self.n_head)
                for _ in range(self.num_hyperbolic_blocks)
            ])

        self.norm_scaler = nn.Parameter(torch.tensor(1.0 / self.hidden_size ** 0.5), requires_grad=config.trainable_norm_scaler)
        self.norm_clip_factor = torch.tensor(config.norm_clip_factor) if config.norm_clip_factor else None

        self.pooling = getattr(config, "pooling", "euc_cls")
        if self.pooling in POOLER_LAYERS:
            pooler_cls = POOLER_LAYERS.get(self.pooling)
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")
        self.pooler = pooler_cls(config, self.manifold) if add_pooling_layer else None

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
        hyp_project=True
    ):
        # clamp norm_scaler to maximum of 1.0
        if self.norm_scaler is not None:
            self.norm_scaler.data = torch.clamp(self.norm_scaler.data, max=1.0)
        # clamp manifold curvature
        self.manifold.k.data = torch.clamp(self.manifold.k.data, **self._curv_minmax)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder_out = self.base_encoder(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        euc_hidden_state = encoder_out.last_hidden_state
        

        if self.delayer_norm is not None:
            euc_hidden_state = self.delayer_norm(euc_hidden_state)

        if "euc" in self.pooling and self.pooler is not None:
            euc_hidden_state = self.pooler(euc_hidden_state, input_ids=input_ids, attention_mask=attention_mask)

        if hyp_project:
        # if False:
            # cast the computation to float32 for numerical stability
            with torch.autocast("cuda", dtype=torch.float32):
                # Project to the hyperbolic hidden size
                hyp_hidden_state = self.projection(euc_hidden_state)
                if self.norm_scaler is not None:
                    hyp_hidden_state = hyp_hidden_state * self.norm_scaler
                

                if self.norm_clip_factor:
                    hidden_norm = hyp_hidden_state.norm(dim=-1, keepdim=True)
                    # Clipping norm of hidden states -- following Clipped HNN
                    hyp_hidden_state = torch.minimum(torch.ones_like(hidden_norm), self.norm_clip_factor/hidden_norm) * hyp_hidden_state

                hyp_hidden_state = self.manifold.expmap0(F.pad(hyp_hidden_state, pad=(1,0), value=0))
                

                for i in range(self.num_hyperbolic_blocks):
                    hyp_hidden_state = self.hyperbolic_blocks[i](hyp_hidden_state)
                
        else:
            hyp_hidden_state = F.normalize(euc_hidden_state, dim=-1)

        if self.pooler is not None and "hyp" in self.pooling:
            pooled_output = self.pooler(hyp_hidden_state, attention_mask=attention_mask)
        else:
            pooled_output = hyp_hidden_state

        if encoder_out.hidden_states is not None:
            hidden_states = encoder_out.hidden_states + (hyp_hidden_state,)
        else:
            hidden_states = None
        

        with torch.no_grad():
            info = {}
            euc_hidden_norm = euc_hidden_state.detach().norm(dim=-1)
            info["euc_hidden_norm"] = euc_hidden_norm.mean()
            info["euc_hidden_norm_std"] = euc_hidden_norm.std()
            hyp_time = hyp_hidden_state[..., 0].detach()
            info["hyperbolic_time"] = hyp_time.mean()
            info["hyperbolic_time_std"] = hyp_time.std()
            info["curvature"] = self.manifold.k.detach()
            info["norm_scaler"] = self.norm_scaler.detach()

        return ELBertOutput(
            euc_hidden_state=euc_hidden_state,
            last_hidden_state=hyp_hidden_state,
            pooler_output=pooled_output if self.pooler is not None else None,
            hidden_states=hidden_states,
            attentions=encoder_out.attentions,
            manifold=self.manifold,
            info=info,
        )

class ELBertForPreTraining(ELBertPreTrainedModel):
    _tied_weights_keys = ["head.decoder.z"]

    def __init__(self, config: GPT2Config, base_pretrained=True):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"

        self.manifold = Lorentz(config.curvature_init, learnable=config.trainable_curvature)

        self.bert = ELBertModel(
            config, manifold=self.manifold,
            base_pretrained=base_pretrained,
            add_pooling_layer=getattr(config, "add_pooling_layer", False)
        )
        self.head = ELBertLMPredictionHead(config, hidden_size=self.bert.hidden_size, manifold=self.manifold)
        self.mlm_loss = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        if config.tie_word_embeddings:
            raise ValueError("Tying word embeddings is not supported for ELBertForPreTraining")

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        If labels are provided, they must be -100 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        input_len = input_ids.shape[-1]

        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None

        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() >= 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx)

        with torch.autocast("cuda", dtype=torch.float32):
            prediction_scores = self.head(sequence_output)

            total_loss = None
            if labels is not None:
                if self.dense_seq_output and labels is not None:  # prediction_scores are already flattened
                    masked_lm_loss = self.mlm_loss(prediction_scores, labels.flatten()[masked_token_idx])
                else:
                    masked_lm_loss = self.mlm_loss(
                        rearrange(prediction_scores, "... v -> (...) v"),
                        rearrange(labels, "... -> (...)"),
                    )
                total_loss = masked_lm_loss.float()

        with torch.no_grad():
            info = {}
            euc_hidden_norm = outputs.euc_hidden_state.detach().norm(dim=-1)
            info["euc_hidden_norm"] = euc_hidden_norm.mean()
            info["euc_hidden_norm_std"] = euc_hidden_norm.std()
            hyp_time = outputs.last_hidden_state[..., 0].detach()
            info["hyperbolic_time"] = hyp_time.mean()
            info["hyperbolic_time_std"] = hyp_time.std()
            info["curvature"] = self.manifold.k.detach()
            info["norm_scaler"] = self.bert.norm_scaler.detach()

        return ELBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            info=info,
        )


class ELBertForSequenceClassification(ELBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"

        self.bert = ELBertModel(config, add_pooling_layer=add_pooling_layer)
        classifier_dropout = getattr(config, "classifier_dropout", config.embd_pdrop)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
