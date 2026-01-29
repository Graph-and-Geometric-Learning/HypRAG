# from contextlib import nullcontext
# from functools import partial

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import hyplib.nn as hnn
# import torch.nn.functional as F
# from flash_attn.bert_padding import pad_input, unpad_input
# from flash_attn.ops.rms_norm import RMSNorm
# from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

# from layers.activations import quick_gelu
# from layers.attention import FlashAttentionPooling
# from layers.block import Block
# from layers.mlp import MLP, GatedMLP
# from models.decoder import DecoderModel
# from models.decoder.gpt_neox import gpt_neox_config_to_gpt2_config
# from models.decoder.open_lm import open_lm_config_to_gpt2_config
# from models.decoder.llama import llama_config_to_gpt2_config
# from models.encoder import NomicBertModel, bert_config_to_nomic_config
# from models.encoder import ALBertModel, ELBertModel, CONFIG_CONVERTER_REGISTRY
# from models.encoder.configuration_albert import ALBertConfig
# from models.encoder import convert_base_model_config_to_elbert_config

# from models.vit import (
#     ViTModel,
#     clip_config_to_vit_config,
#     dino_config_to_vit_config,
#     hf_vit_config_to_vit_config,
#     timm_name_to_vit_config,
# )

# def update_model_config(model_config, config):
#     for key, value in config.__dict__.items():
#         if key.startswith("_"):
#             continue
#         if hasattr(model_config, key):
#             try:
#                 if model_config.__dict__[key] != value:
#                     print(f"Setting {key} to {value}")
#                     setattr(model_config, key, value)
#             except KeyError:
#                     print(f"Setting {key} to {value}")
#                     setattr(model_config, key, value)
#     return model_config


# class LogitScale(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.logit_scale = nn.Parameter(
#             torch.ones([]) * np.log(config.logit_scale), requires_grad=config.trainable_logit_scale
#         )

#     def forward(self, x):
#         return x * self.logit_scale.exp()

#     def __repr__(self):
#         return f"LogitScale(logit_scale={self.logit_scale.exp().item()}, trainable={self.logit_scale.requires_grad})"


# class ClsSelector(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, hidden_states, input_ids, attention_mask):
#         return hidden_states[:, 0]


# class LastTokenPooling(nn.Module):
#     def __init__(self, eos_token_id):
#         super().__init__()
#         self.eos_token_id = eos_token_id

#     def forward(self, hidden_states, input_ids, attention_mask):
#         # get the embedding corresponding to the first eos token
#         # we don't substract 1 because the eos token is already included in the input_ids and attention_mask
#         # and we want to get the embedding of the last token
#         sequence_lengths = attention_mask.sum(-1) - 1
#         selected_tokens = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths]

#         if not torch.all(selected_tokens == self.eos_token_id):
#             raise ValueError(
#                 f"The last token of the input_ids is not the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
#             )
#         prev_token = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths - 1]
#         if torch.any(prev_token == self.eos_token_id):
#             raise ValueError(
#                 f"The second to last token of the input_ids is the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
#             )

#         embs = hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

#         return embs


# class MeanPooling(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, hidden_states, input_ids, attention_mask):
#         if attention_mask is None:
#             # for vit, no attention mask is provided
#             return torch.mean(hidden_states, dim=1)

#         s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
#         d = attention_mask.sum(axis=1, keepdim=True).float()
#         return s / d


# class MultiHeadAttentionPooling(nn.Module):
#     def __init__(self, config):
#         # adapted from https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/models/vit.py#L158
#         super().__init__()
#         self.attn = FlashAttentionPooling(config)
#         activation = (
#             F.sigmoid
#             if config.activation_function == "glu"
#             else (
#                 F.silu
#                 if config.activation_function == "swiglu"
#                 else (quick_gelu if config.activation_function == "quick_gelu" else F.gelu)
#             )
#         )
#         if config.activation_function in ["glu", "swiglu"]:
#             self.mlp = GatedMLP(
#                 config.n_embd,
#                 hidden_features=config.n_inner,
#                 bias1=config.mlp_fc1_bias,
#                 bias2=config.mlp_fc2_bias,
#                 activation=activation,
#                 fused_bias_fc=config.fused_bias_fc,
#             )
#         else:
#             self.mlp = MLP(
#                 config.n_embd,
#                 hidden_features=config.n_inner,
#                 bias1=config.mlp_fc1_bias,
#                 bias2=config.mlp_fc2_bias,
#                 activation=activation,
#                 fused_bias_fc=config.fused_bias_fc,
#             )
#         norm_cls = partial(
#             nn.LayerNorm if not config.use_rms_norm else RMSNorm,
#             eps=config.layer_norm_epsilon,
#         )
#         self.norm1 = norm_cls(config.n_embd)

#     def forward(self, hidden_states, input_ids, attention_mask):
#         if attention_mask is not None:
#             hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
#         else:
#             indices = None
#             cu_seqlens = None
#             max_seqlen_in_batch = None

#         attn_outputs = self.attn(
#             hidden_states,
#             attention_mask=attention_mask,
#             is_padded_inputs=True,
#             cu_seqlens_k=cu_seqlens,
#             max_seqlen_k=max_seqlen_in_batch,
#         )

#         normed = self.norm1(attn_outputs)
#         hidden_states = hidden_states + self.mlp(normed)
#         if attention_mask is not None:
#             hidden_states = pad_input(hidden_states, indices, cu_seqlens, max_seqlen_in_batch)

#         return hidden_states[:, 0]


# class HypBiEncoder(PreTrainedModel):
#     _supports_flash_attn_2 = True
#     _supports_flex_attn = True

#     def __init__(self, config):
#         super().__init__(config)

#         if config.use_fused_kernels:
#             print(f"Initializing {config.model_name}, pretrained={config.pretrained}")
#             # set default to true for backward compatibility with old models?
#             if "elbert" in config.model_type:
#                 model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

#                 if model_config.model_type == "elbert":
#                     self.trunk = ELBertModel.from_pretrained(
#                         config.model_name,
#                         add_pooling_layer=True,
#                         config=model_config,
#                     )
#                     if config.pretrained:
#                         print(f"Loading weights from {config.model_name}")

#                         state = torch.load(
#                             os.path.join(config.model_name, "pytorch_model.bin"),
#                             map_location="cpu",
#                         )

#                         state_dict = state.get("state_dict", state)

#                         trunk_state_dict = {
#                             k[len("trunk."):]: v
#                             for k, v in state_dict.items()
#                             if k.startswith("trunk.")
#                         }

#                         missing, unexpected = self.trunk.load_state_dict(
#                             trunk_state_dict,
#                             strict=False,
#                             assign=True,
#                         )
                        
#                 else:
#                     model_config = convert_base_model_config_to_elbert_config(model_config, config)
#                     self.trunk = ELBertModel(
#                         config=model_config,
#                         add_pooling_layer=True,
#                         base_pretrained=config.pretrained,
#                     )
#             elif "albert" in config.model_type:
#                 model_config = ALBertConfig.from_pretrained(config.model_name, trust_remote_code=True)
#                 config.attn_implementation = "flash_attention_2"
#                 model_config = update_model_config(model_config, config)
                
#                 self.trunk = ALBertModel(
#                     config=model_config,
#                     add_pooling_layer=True,
#                 )
#                 if config.pretrained:
#                     print(f"Loading weights from {config.model_name}")

#                     state = torch.load(
#                         os.path.join(config.model_name, "pytorch_model.bin"),
#                         map_location="cpu",
#                     )

#                     state_dict = state.get("state_dict", state)

#                     trunk_state_dict = {
#                         k[len("trunk."):]: v
#                         for k, v in state_dict.items()
#                         if k.startswith("trunk.")
#                     }

#                     missing, unexpected = self.trunk.load_state_dict(
#                         trunk_state_dict,
#                         strict=False,
#                         assign=True,
#                     )

#         else:
#             raise ValueError(f"Model type {config.model_type} not supported")

#         if config.freeze:
#             self.trunk.eval()
#             for param in self.trunk.parameters():
#                 param.requires_grad = False

#             self.frozen_trunk = True
#         else:
#             self.frozen_trunk = False

#         if config.gradient_checkpointing:
#             self.trunk.gradient_checkpointing_enable({"use_reentrant": False})

#         if config.projection_dim:
#             self.proj = hnn.LorentzLinear(self.trunk.config.hidden_size, config.projection_dim)
#         else:
#             self.proj = nn.Identity()

#     @property
#     def manifold(self):
#         return self.trunk.manifold

#     def forward(self, input_ids, attention_mask=None, is_padded_inputs=True, **kwargs):
#         context = torch.no_grad if self.frozen_trunk else nullcontext
#         with context():
#             trunk_output = self.trunk(input_ids, attention_mask=attention_mask, **kwargs)

#         manifold = self.trunk.manifold

#         embedding = trunk_output.pooler_output
#         embedding = self.proj(embedding)
#         info = trunk_output.info

#         return {"embedding": embedding, "manifold": manifold, "info": info}

from contextlib import nullcontext
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import hyplib.nn as hnn
import torch.nn.functional as F
from flash_attn.bert_padding import pad_input, unpad_input
# from flash_attn.ops.rms_norm import RMSNorm
from torch.nn import RMSNorm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from layers.activations import quick_gelu
from layers.attention import FlashAttentionPooling
from layers.block import Block
from layers.mlp import MLP, GatedMLP
from models.encoder import ALBertModel, ELBertModel, CONFIG_CONVERTER_REGISTRY, ALBertConfig
from models.encoder import convert_base_model_config_to_elbert_config


def update_model_config(model_config, config):
    model_config.model_type = 'albert'
    for key, value in config.__dict__.items():
        if key.startswith("_"):
            continue
        if hasattr(model_config, key) and model_config.__dict__[key] != value and value is not None:
            print(f"Setting {key} to {value}")
            setattr(model_config, key, value)
    return model_config


class LogitScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(config.logit_scale), requires_grad=config.trainable_logit_scale
        )

    def forward(self, x):
        return x * self.logit_scale.exp()

    def __repr__(self):
        return f"LogitScale(logit_scale={self.logit_scale.exp().item()}, trainable={self.logit_scale.requires_grad})"


class ClsSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        return hidden_states[:, 0]


class LastTokenPooling(nn.Module):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id

    def forward(self, hidden_states, input_ids, attention_mask):
        # get the embedding corresponding to the first eos token
        # we don't substract 1 because the eos token is already included in the input_ids and attention_mask
        # and we want to get the embedding of the last token
        sequence_lengths = attention_mask.sum(-1) - 1
        selected_tokens = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths]

        if not torch.all(selected_tokens == self.eos_token_id):
            raise ValueError(
                f"The last token of the input_ids is not the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )
        prev_token = input_ids[torch.arange(input_ids.shape[0]), sequence_lengths - 1]
        if torch.any(prev_token == self.eos_token_id):
            raise ValueError(
                f"The second to last token of the input_ids is the eos token: {selected_tokens}\n{input_ids}\n{sequence_lengths}"
            )

        embs = hidden_states[torch.arange(hidden_states.shape[0]), sequence_lengths]

        return embs


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is None:
            # for vit, no attention mask is provided
            return torch.mean(hidden_states, dim=1)

        s = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(axis=1, keepdim=True).float()
        return s / d


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, config):
        # adapted from https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/models/vit.py#L158
        super().__init__()
        self.attn = FlashAttentionPooling(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (
                F.silu
                if config.activation_function == "swiglu"
                else (quick_gelu if config.activation_function == "quick_gelu" else F.gelu)
            )
        )
        if config.activation_function in ["glu", "swiglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)

    def forward(self, hidden_states, input_ids, attention_mask):
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        else:
            indices = None
            cu_seqlens = None
            max_seqlen_in_batch = None

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            is_padded_inputs=True,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=max_seqlen_in_batch,
        )

        normed = self.norm1(attn_outputs)
        hidden_states = hidden_states + self.mlp(normed)
        if attention_mask is not None:
            hidden_states = pad_input(hidden_states, indices, cu_seqlens, max_seqlen_in_batch)

        return hidden_states[:, 0]


class HypBiEncoder(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_flex_attn = True

    def __init__(self, config):
        super().__init__(config)

        if config.use_fused_kernels:
            print(f"Initializing {config.model_name}, pretrained={config.pretrained}")
            # set default to true for backward compatibility with old models?
            if "elbert" in config.model_type:
                model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

                if model_config.model_type == "elbert":
                    if config.pretrained:
                        state_dict = torch.load(os.path.join(config.model_name, 'pytorch_model.bin'), map_location="cpu")
                        # trim state_dict if 'trunk.' prefix exists
                        if any(key.startswith("trunk.") for key in state_dict.keys()):
                            state_dict = {key[len("trunk.") :]: value for key, value in state_dict.items()}
                        self.trunk = ELBertModel.from_pretrained(
                            config.model_name,
                            add_pooling_layer=True,
                            config=model_config,
                        )
                        self.trunk.load_state_dict(state_dict, strict=False)
                    else:
                        self.trunk = ELBertModel(
                            config=model_config,
                            add_pooling_layer=True,
                        )
                else:
                    model_config = convert_base_model_config_to_elbert_config(model_config, config)
                    self.trunk = ELBertModel(
                        config=model_config,
                        add_pooling_layer=True,
                        base_pretrained=config.pretrained,
                    )
            elif "albert" in config.model_type:
                model_config = ALBertConfig.from_pretrained(config.model_name, trust_remote_code=True)
                config.attn_implementation = "flash_attention_2"
                model_config = update_model_config(model_config, config)
                
                self.trunk = ALBertModel(
                    config=model_config,
                    add_pooling_layer=True,
                )
                if config.pretrained:
                    print(f"Loading weights from {config.model_name}")

                    state = torch.load(
                        os.path.join(config.model_name, "pytorch_model.bin"),
                        map_location="cpu",
                    )

                    state_dict = state.get("state_dict", state)

                    trunk_state_dict = {
                        k[len("trunk."):]: v
                        for k, v in state_dict.items()
                        if k.startswith("trunk.")
                    }
                    missing, unexpected = self.trunk.load_state_dict(
                        trunk_state_dict,
                        strict=False,
                        assign=True,
                    )
        else:
            raise ValueError(f"Model type {config.model_type} not supported")

        if config.freeze:
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad = False

            self.frozen_trunk = True
        else:
            self.frozen_trunk = False

        if config.gradient_checkpointing:
            self.trunk.gradient_checkpointing_enable({"use_reentrant": False})

        if config.projection_dim:
            self.proj = hnn.LorentzLinear(self.trunk.config.hidden_size, config.projection_dim)
        else:
            self.proj = nn.Identity()

    @property
    def manifold(self):
        return self.trunk.manifold

    def save_pretrained(self, save_directory, **kwargs):
        self.trunk.save_pretrained(save_directory, **kwargs)

    def load_pretrained(self, pretrained_model_name_or_path, **kwargs):
        # print("Loading using load_pretrained")
        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        # attn_implementation = self.trunk.config.attn_implementation
        # self.trunk = self.trunk.from_pretrained(
        #     pretrained_model_name_or_path,
        #     config=config,
        #     **kwargs
        # )
        # self.trunk.config.attn_implementation = attn_implementation
        # return self
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        self.trunk.load_state_dict(torch.load(weights_path), **kwargs)
    def forward(self, input_ids, attention_mask=None, return_cone_info=False, **kwargs):
        context = torch.no_grad if self.frozen_trunk else nullcontext
        with context():
            trunk_output = self.trunk(input_ids, attention_mask=attention_mask, **kwargs)

        manifold = self.trunk.manifold

        embedding = trunk_output.pooler_output
        embedding = self.proj(embedding)

        if return_cone_info:
            hidden_states = trunk_output.hidden_states
            # randomly select a substring of the hidden states and then run pooling on it
            min_length = int(0.6 * hidden_states.shape[1])
            length = torch.randint(low=min_length, high=hidden_states.shape[1] + 1, size=(1,)).item()
            start_idx = torch.randint(low=0, high=hidden_states.shape[1] - length + 1, size=(1,)).item()
            sub_hidden_states = hidden_states[:, start_idx : start_idx + length, :]
            sub_pooled_output = self.trunk.pooler(sub_hidden_states)
            sampled_embedding = self.proj(sub_pooled_output)
        else:
            sampled_embedding = None

        info = trunk_output.info

        return {
            "embedding": embedding,
            "manifold": manifold,
            "sub_embedding": sampled_embedding,
            "info": info,
        }