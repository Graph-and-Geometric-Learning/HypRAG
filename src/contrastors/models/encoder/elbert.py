import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import BertConfig, GPT2Config, PretrainedConfig, ModernBertConfig, Qwen3Config

from .configuration_nomic_bert import NomicBertConfig
from .configuration_elbert import ELBertConfig


__all__ = [
    "convert_base_model_config_to_elbert_config",
    "get_base_hidden_size",
]

def convert_base_model_config_to_elbert_config(config: PretrainedConfig, new_config) -> ELBertConfig:
    """
    Update the ELBert configuration based on the model name.
    """
    if config.model_type != "elbert":
        elbert_config = ELBertConfig(base_model_name=new_config.model_name, base_model_config=config)
    else:
        elbert_config = config

    elbert_config.activation_function = get_activation_function(config)

    if hasattr(new_config, "curvature_init") and new_config.curvature_init is not None:
        elbert_config.curvature_init = new_config.curvature_init
    if hasattr(new_config, "curvature_max") and new_config.curvature_max is not None:
        elbert_config.curvature_max = new_config.curvature_max
    if hasattr(new_config, "curvature_min") and new_config.curvature_min is not None:
        elbert_config.curvature_min = new_config.curvature_min
    if hasattr(new_config, "trainable_curvature") and new_config.trainable_curvature is not None:
        elbert_config.trainable_curvature = new_config.trainable_curvature
    if hasattr(new_config, "projection") and new_config.projection is not None:
        elbert_config.projection = new_config.projection
    if hasattr(new_config, "num_hyperbolic_blocks") and new_config.num_hyperbolic_blocks is not None:
        elbert_config.num_hyperbolic_blocks = new_config.num_hyperbolic_blocks
    if hasattr(new_config, "hidden_size") and new_config.hidden_size is not None:
        elbert_config.hidden_size = new_config.hidden_size
    if hasattr(new_config, "norm_scaler") and new_config.norm_scaler is not None:
        elbert_config.norm_scaler = new_config.norm_scaler
    if hasattr(new_config, "trainable_norm_scaler") and new_config.trainable_norm_scaler is not None:
        elbert_config.trainable_norm_scaler = new_config.trainable_norm_scaler
    if hasattr(new_config, "norm_clip_factor") and new_config.norm_clip_factor is not None:
        elbert_config.norm_clip_factor = new_config.norm_clip_factor
    if hasattr(new_config, "pooling") and new_config.pooling is not None:
        elbert_config.pooling = new_config.pooling
    if hasattr(new_config, "delayer_norm") and new_config.delayer_norm is not None:
        elbert_config.delayer_norm = new_config.delayer_norm
    if hasattr(new_config, "initializer_range") and new_config.initializer_range is not None:
        elbert_config.initializer_range = new_config.initializer_range
    if hasattr(new_config, "tie_word_embeddings") and new_config.tie_word_embeddings is not None:
        elbert_config.tie_word_embeddings = new_config.tie_word_embeddings

    return elbert_config


def get_base_hidden_size(config: PretrainedConfig) -> int:
    """
    Get the base embedding dimension based on the model type.
    """
    if isinstance(config, (BertConfig, ModernBertConfig)):
        return config.hidden_size
    elif isinstance(config, GPT2Config):
        return config.n_embd
    elif isinstance(config, NomicBertConfig):
        return config.embedding_dim
    elif isinstance(config, Qwen3Config):
        return config.hidden_size
    elif isinstance(config, dict):
        if "n_embd" in config.keys():
            return config['n_embd']
        else:
            return config['hidden_size']
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

def get_activation_function(config: PretrainedConfig):
    """
    Get the activation function based on the configuration.
    """
    if hasattr(config, "hidden_act"):
        activation = config.hidden_act
    elif hasattr(config, "hidden_activation"):
        activation = config.hidden_activation
    elif hasattr(config, "activation_function"):
        activation = config.activation_function
    else:
        raise ValueError("Activation function not specified in the configuration.")

    return activation
