import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import BertConfig, GPT2Config, PretrainedConfig, ModernBertConfig

from .configuration_nomic_bert import NomicBertConfig
from .configuration_albert import ALBertConfig

__all__ = [
    "bert_config_to_albert_config",
    "albert_config_to_bert_config",
    "albert_config_to_modernbert_config",
    "modernbert_config_to_albert_config",
]


def bert_config_to_albert_config(bert_config: BertConfig) -> ALBertConfig:
    return ALBertConfig(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        hidden_act=bert_config.hidden_act,
        hidden_dropout_prob=bert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
        max_position_embeddings=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        initializer_range=bert_config.initializer_range,
        layer_norm_eps=bert_config.layer_norm_eps,
        pad_token_id=bert_config.pad_token_id,
        position_embedding_type=bert_config.position_embedding_type,
        use_cache=bert_config.use_cache,
        # These are new arguments not in the original GPT2Config
        manifold=1.0,
        rope_theta=10000.0,
        pooling="cls",
        norm_layer="layer_norm",
        decoder_bias=False,
        classifier_bias=True,
        classifier_activation="tanh",
    )


def albert_config_to_bert_config(albert_config: ALBertConfig) -> BertConfig:
    return BertConfig(
        vocab_size=albert_config.vocab_size,
        hidden_size=albert_config.hidden_size,
        num_hidden_layers=albert_config.num_hidden_layers,
        num_attention_heads=albert_config.num_attention_heads,
        intermediate_size=albert_config.intermediate_size,
        hidden_act=albert_config.hidden_act,
        hidden_dropout_prob=albert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=albert_config.attention_probs_dropout_prob,
        max_position_embeddings=albert_config.max_position_embeddings,
        type_vocab_size=albert_config.type_vocab_size,
        initializer_range=albert_config.initializer_range,
        layer_norm_eps=albert_config.layer_norm_eps,
        pad_token_id=albert_config.pad_token_id,
        position_embedding_type=albert_config.position_embedding_type,
        use_cache=albert_config.use_cache,
    )


def modernbert_config_to_albert_config(gpt2_config: ModernBertConfig) -> ALBertConfig:
    raise NotImplementedError()

def albert_config_to_modernbert_config(albert_config: ALBertConfig) -> ModernBertConfig:
    raise NotImplementedError()
