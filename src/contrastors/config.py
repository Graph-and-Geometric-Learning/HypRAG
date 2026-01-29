from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dataset.constants import OPENAI_IMAGE_DATASET_MEAN, OPENAI_IMAGE_DATASET_STD
from models import MODEL_REGISTRY


class TrainArgs(BaseModel):
    trainer_type: str
    num_epochs: int
    num_train_steps: Optional[int] = None
    optimizer: str = "adamw"
    learning_rate: float
    weight_decay: float
    eps: Optional[float] = 1e-8
    warmup_steps: Optional[int] = None
    warmup_pct: Optional[float] = None
    cooldown_steps: Optional[int] = None
    checkpoint: Optional[str] = None
    wandb: bool
    wandb_project_name: str
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_mode: Optional[str] = None
    log_grads_every: int
    log_lr_every: int
    save_every: Optional[int] = None
    save_total_limit: Optional[int] = None
    eval_steps: Optional[int] = None
    eval_strategy: Optional[str] = None
    output_dir: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = 1
    # if using deepspeed, this will be ignored
    schedule_type: str
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    loss_fn: Optional[str] = None
    grad_cache: Optional[bool] = None
    chunk_size: Optional[int] = None
    clamp_logits: Optional[bool] = True
    logit_max: Optional[float] = 100.0
    add_l2_loss: Optional[bool] = False
    matryoshka_dims: Optional[List[int]] = None
    matryoshka_loss_weights: Optional[List[float]] = None
    model_config = ConfigDict(validate_assignment=True)
    profile: Optional[bool] = False
    router_aux_loss_coef: Optional[float] = 0.001
    distill_loss_fn: Optional[str] = "mse"
    distill_temperature: Optional[float] = 1.0
    dtype: Optional[str] = "fp32"
    distance: Optional[str] = "euclidean"
    eval_distance: Optional[str] = "euclidean"

    @field_validator('logit_max')
    @classmethod
    def set_logic_max(cls, max):
        return max or 100.0

    @field_validator("eval_strategy")
    @classmethod
    def validate_eval_strategy(cls, strategy):
        if strategy not in ["steps", "epochs", "no"]:
            raise ValueError(f"Eval strategy {strategy} not found in eval strategy registry")
        return strategy

    @model_validator(mode="after")
    def validate_steps_set(self):
        # validate that eval_steps is set if eval_strategy is set to steps
        eval_steps, eval_strategy = self.eval_steps, self.eval_strategy
        if eval_strategy == "steps" and eval_steps is None:
            raise ValueError("Eval steps must be set if eval strategy is set to steps")

        return self

    @model_validator(mode="after")
    def validate_matryoshka_no_grad_cache(self):
        # validate that matryoska isn't set if grad_cache is set
        matryoshka, grad_cache = self.matryoshka_dims, self.grad_cache
        if matryoshka is not None and grad_cache:
            raise ValueError("Matryoshka dims cannot be set if grad cache is set")

        return self

    @field_validator("distill_loss_fn")
    @classmethod
    def validate_distill_loss_fn(cls, distill_loss_fn):
        if distill_loss_fn not in ["mse", "kd", "towers", "stella"]:
            raise ValueError(f"Distill loss fn {distill_loss_fn} not found in distill loss fn registry")
        return distill_loss_fn

    @field_validator('trainer_type')
    @classmethod
    def validate_trainer_type(cls, trainer_type):
        if trainer_type not in ["encoder", "hyp_encoder", "mlm", "glue", "image_text", "locked_text", "mmlm", "distill"]:
            raise ValueError(f"Trainer type {trainer_type} not found in trainer registry")
        return trainer_type


class DataArgs(BaseModel):
    data_path: Optional[str] = '/tmp'
    shuffle: bool
    workers: int
    batch_size: int
    seed: int
    val_pct: Optional[float] = None


class MLMDataArgs(DataArgs):
    dataset: Optional[str] = None
    lang: Optional[str] = None
    tokenized_dataset: Optional[str] = None
    mlm_prob: Optional[float] = None
    eval_batch_size: Optional[int] = None
    task_name: Optional[Optional[str]] = None
    val_mlm_prob: Optional[float] = None

    @model_validator(mode="after")
    def validate_data(self):
        tokenized, task_name = self.tokenized_dataset, self.task_name
        if tokenized is None and task_name is None:
            raise ValueError("Either tokenized dataset or task name must be set")
        return self

    @model_validator(mode="after")
    def validate_mlm(self):
        tokenized, mlm_prob, val_prob = (
            self.tokenized_dataset,
            self.mlm_prob,
            self.val_mlm_prob,
        )
        # validate mlm_prob if tokenized is set
        if tokenized is not None and mlm_prob is None:
            raise ValueError("MLM probability must be set if tokenized dataset is set")
        if tokenized is not None and val_prob is None:
            raise ValueError("Validation MLM probability must be set if tokenized dataset is set")
        if mlm_prob is not None and (mlm_prob < 0 or mlm_prob > 1):
            raise ValueError("MLM probability must be between 0 and 1")
        if val_prob is not None and (val_prob < 0 or val_prob > 1):
            raise ValueError("Validation MLM probability must be between 0 and 1")
        return self


class ContrastiveDataArgs(DataArgs):
    input_shards: str
    download: Optional[bool] = False
    process_one_shard: Optional[bool] = False
    streaming: Optional[str] = None
    shuffle_buffer_size: Optional[int] = 5000000
    weighted_sampling: Optional[bool] = False
    verbose: Optional[bool] = False
    seq_len: Optional[int] = None
    sample_negatives: Optional[bool] = True
    query_max_length: Optional[int] = None
    document_max_length: Optional[int] = None


class ImageTextDataArgs(DataArgs):
    image_text_shards: str
    eval_batch_size: int
    imagenet_val_path: Optional[str] = None
    eval_flickr: Optional[bool] = False
    train_num_samples: Optional[int] = None
    dataset_resampled: Optional[bool] = False
    mlm_prob: Optional[float] = None

# TODO: should we just make text and image nested of model_args?
class ModelArgs(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()  # disable all protected‐namespace checks
    )

    model_type: str
    pretrained: Optional[bool] = True
    checkpoint: Optional[str] = None
    tokenizer_name: str
    gradient_checkpointing: Optional[bool] = False
    num_hidden_layers: Optional[int] = 12
    num_attention_heads: Optional[int] = 12
    intermediate_size: Optional[int] = 12
    n_embd: Optional[int] = 768
    pretrained_weights: Optional[str] = None
    tie_word_embeddings: Optional[bool] = True
    logit_scale: Optional[float] = 1 / 0.07
    trainable_logit_scale: Optional[bool] = False
    num_negatives: Optional[int] = 7
    add_prefix: Optional[bool] = False
    add_eos: Optional[bool] = False
    query_prefix: Optional[str] = "search_query: "
    document_prefix: Optional[str] = "search_document: "
    pooling: Optional[str] = None
    gradient_checkpointing: Optional[bool] = False

    @field_validator('logit_scale')
    @classmethod
    def set_logit_scale(cls, scale):
        return scale or 1 / 0.07

class ALBertModelArgs(ModelArgs):
    model_type: str = "albert"
    model_name: str
    tokenizer_name: Optional[str] = None
    seq_len: Optional[int] = 2048
    hidden_size: Optional[int] = None
    rope_theta: Optional[float] = 10000.0
    norm_layer: Optional[str] = "layer_norm"
    use_rotary_embeddings: Optional[bool] = True
    attn_implementation: Optional[str] = "flex_attention"
    trainable_curvature: Optional[bool] = False
    curvature_init: Optional[float] = 1.0

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "albert":
            raise ValueError(f"Expected model type 'albert', but got {model_type}")
        return model_type


class ALBertPretrainingModelArgs(ALBertModelArgs):
    model_type: str = "albert_pretraining"
    model_name: str

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "albert_pretraining":
            raise ValueError(f"Expected model type 'albert_pretraining', but got {model_type}")
        return model_type


class NomicModelArgs(ModelArgs):
    model_type: str = "nomic_bert"
    seq_len: Optional[int] = None
    rotary_emb_fraction: Optional[float] = None
    rotary_emb_base: Optional[int] = 10_000
    pad_vocab_to_multiple_of: Optional[int] = None
    use_rms_norm: Optional[bool] = None
    model_name: Optional[str] = None
    nomic_encoder: Optional[bool] = False
    tokenizer_name: Optional[str] = None
    activation_function: Optional[str] = "gelu"
    qkv_proj_bias: Optional[bool] = True
    mlp_fc1_bias: Optional[bool] = True
    mlp_fc2_bias: Optional[bool] = True
    attn_pdrop: Optional[float] = 0.0
    projection_dim: Optional[int] = None
    freeze: Optional[bool] = False
    precomputed: Optional[bool] = False
    hamming: Optional[bool] = False
    ema: Optional[bool] = False
    patch_dropout: Optional[float] = 0.0
    num_experts: Optional[int] = 0
    moe_top_k: Optional[int] = 1
    ffn_div: Optional[int] = 1 # 1 for no fine grained expert segmentation
    moe_normalize_expert_weights: Optional[bool] = False
    expert_choice_router: Optional[bool] = False
    num_shared_experts: Optional[int] = 0
    resid_pdrop: Optional[float] = None
    moe_every_n_layers: Optional[int] = 0
    distill_init_pretrained: Optional[bool] = False


    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "nomic_bert":
            raise ValueError(f"Expected model type 'nomic_bert', but got {model_type}")
        return model_type


class ELBertModelArgs(ModelArgs):
    model_type: str = "elbert"
    model_name: str
    curvature_init: Optional[float] = None
    curvature_max: Optional[float] = None
    curvature_min: Optional[float] = None
    trainable_curvature: Optional[bool] = None
    projection: Optional[str] = None
    num_hyperbolic_blocks: Optional[int] = 0
    hidden_size: Optional[int] = None
    seq_len: Optional[int] = None
    trainable_norm_scaler: Optional[bool] = None
    norm_clip_factor: Optional[float] = None
    delayer_norm: Optional[bool] = None

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "elbert":
            raise ValueError(f"Expected model type 'elbert', but got {model_type}")
        return model_type

class ELBertPretrainingModelArgs(ELBertModelArgs):
    model_type: str = "elbert_pretraining"
    model_name: str

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "elbert_pretraining":
            raise ValueError(f"Expected model type 'elbert_pretraining', but got {model_type}")
        return model_type


class ModernBertModelArgs(ModelArgs):
    model_type: str = "modernbert"
    model_name: str
    hidden_size: Optional[int] = None
    seq_len: Optional[int] = None

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, model_type):
        if model_type != "modernbert":
            raise ValueError(f"Expected model type 'modernbert', but got {model_type}")
        return model_type


class AugmentationCfg(BaseModel):
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None


class TransformsConfig(BaseModel):
    image_size: Union[int, Tuple[int, int]] = 224
    mean: Optional[Union[float, Tuple[float, float, float]]] = OPENAI_IMAGE_DATASET_MEAN
    std: Optional[Union[float, Tuple[float, float, float]]] = OPENAI_IMAGE_DATASET_STD
    resize_longest_max: bool = False
    fill_color: int = 0
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None


class Config(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()  # disable all protected‐namespace checks
    )

    train_args: TrainArgs
    data_args: Optional[
        Union[
            MLMDataArgs,
            ImageTextDataArgs,
            ContrastiveDataArgs,
        ]
    ] = None
    model_args: Optional[
        Union[
            NomicModelArgs,
            ELBertModelArgs,
            ModernBertModelArgs,
            ALBertModelArgs,
            ELBertPretrainingModelArgs,
            ALBertPretrainingModelArgs,
        ]
    ] = None

    deepspeed: Optional[bool] = False
    deepspeed_config: Optional[dict] = None
    local_rank: Optional[int] = None

    # text_data_args: Optional[ContrastiveDataArgs] = None
    # text_model_args: Optional[ModelArgs] = None
    # vision_model_args: Optional[ModelArgs] = None
    # tower_model_args: Optional[ModelArgs] = None
    # transforms: Optional[TransformsConfig] = None
