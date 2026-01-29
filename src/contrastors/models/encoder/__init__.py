import torch
import torch.distributed as dist

from transformers import AutoTokenizer, BertConfig
from transformers import AutoConfig, AutoModel

from .bert import *
from .albert import *
from .elbert import *
from .configuration_elbert import ELBertConfig
from .configuration_albert import ALBertConfig
from .configuration_nomic_bert import NomicBertConfig

from .modeling_nomic_bert import *
from .modeling_albert import *
from .modeling_elbert import *
from ..model_utils import load_tokenizer


def load_nomic_bert_pretraining(config):
    tokenizer = load_tokenizer(config)
    # model_name = config.model_name
    # config = AutoConfig.from_pretrained(config.model_name)
    # config.model_name = model_name
    hf_config = BertConfig.from_pretrained(config.model_name)
    if hf_config.vocab_size != len(tokenizer):
        print(f"Resizing model vocab from {hf_config.vocab_size} to {len(tokenizer)}")
        hf_config.vocab_size = len(tokenizer)

    hf_config.max_position_embeddings = config.seq_len
    hf_config.rotary_emb_fraction = config.rotary_emb_fraction
    hf_config.rotary_emb_base = config.rotary_emb_base
    hf_config.hidden_size = config.n_embd
    hf_config.num_hidden_layers = config.num_hidden_layers
    hf_config.num_attention_heads = config.num_attention_heads
    hf_config.pad_vocab_to_multiple_of = config.pad_vocab_to_multiple_of
    # use rmsnorm instead of layernorm
    hf_config.use_rms_norm = config.use_rms_norm
    hf_config.hidden_act = config.activation_function
    hf_config.qkv_proj_bias = config.qkv_proj_bias
    hf_config.mlp_fc1_bias = config.mlp_fc1_bias
    hf_config.mlp_fc2_bias = config.mlp_fc2_bias
    hf_config.attention_probs_dropout_prob = config.attn_pdrop

    model_config = bert_config_to_nomic_config(hf_config)
    model_config.tie_word_embeddings = config.tie_word_embeddings
    model = NomicBertForPreTraining(model_config)

    if config.pretrained_weights is not None:
        print(f"Loading pretrained weights from {config.pretrained_weights}")
        model = model.from_pretrained(config.pretrained_weights, config=model_config)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return {"model": model, "tokenizer": tokenizer, "config": model_config}



def load_elbert_pretraining(config):

    tokenizer = load_tokenizer(config)

    base_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    if base_config.vocab_size != len(tokenizer):
        print(f"Resizing model vocab from {base_config.vocab_size} to {len(tokenizer)}")
        base_config.vocab_size = len(tokenizer)

    model_config = convert_base_model_config_to_elbert_config(base_config, config)
    model = ELBertForPreTraining(model_config, base_pretrained=config.pretrained)

    if config.pretrained_weights is not None:
        print(f"Loading pretrained weights from {config.pretrained_weights}")
        model = model.from_pretrained(config.pretrained_weights, config=model_config)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return {"model": model, "tokenizer": tokenizer, "config": model_config}


def load_albert_pretraining(config):
    tokenizer = load_tokenizer(config)

    if config.pretrained_weights is not None:
        print(f"Loading pretrained weights from {config.pretrained_weights}")
        model = ALBertForPreTraining.from_pretrained(config.pretrained_weights)
    else:
        model_config = ALBertConfig.from_pretrained(config.model_name)
        print("====================== in load_albert_pretraining")
        print(sorted(config.__dict__.keys()))
        if model_config.vocab_size != len(tokenizer):
            print(f"Resizing model vocab from {model_config.vocab_size} to {len(tokenizer)}")
            model_config.vocab_size = len(tokenizer)        

        model_config.max_position_embeddings = config.seq_len

        for attr, value in config.__dict__.items():
            if hasattr(model_config, attr) and value is not None:
                print(f"Setting {attr} to {value}")
                setattr(model_config, attr, value)

        model = ALBertForPreTraining(model_config)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return {"model": model, "tokenizer": tokenizer, "config": model_config}


ENCODER_REGISTRY = {
    "nomic_bert": NomicBertForPreTraining,
    "albert_pretraining": ALBertForPreTraining,
    "elbert_pretraining": ELBertForPreTraining,
}


ENCODER_LOADER_REGISTRY = {
    "nomic_bert": load_nomic_bert_pretraining,
    "albert_pretraining": load_albert_pretraining,
    "elbert_pretraining": load_elbert_pretraining,
}

CONFIG_CONVERTER_REGISTRY = {
    "bert_to_nomic_bert": bert_config_to_nomic_config,
    "bert_to_albert": bert_config_to_albert_config,
    "modernbert_to_albert": modernbert_config_to_albert_config,
    "modernbert_to_nomic_bert": modernbert_config_to_nomic_config,

    "albert_to_bert": albert_config_to_bert_config,
    "albert_to_modernbert": albert_config_to_modernbert_config,

    "nomic_bert_to_bert": nomic_config_to_bert_config,
}

AutoConfig.register("elbert", ELBertConfig)
AutoModel.register(ELBertConfig, ELBertModel)

AutoConfig.register("Albert", ALBertConfig)
AutoModel.register(ALBertConfig, ALBertModel)

AutoConfig.register("nomic_bert", NomicBertConfig)
AutoModel.register(NomicBertConfig, NomicBertModel)
