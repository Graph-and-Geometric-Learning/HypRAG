from transformers.configuration_utils import PretrainedConfig


class HypBiEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_type="elbert",
        model_name="EleutherAI/pythia-1b",
        projection_dim=None,
        logit_scale=1 / 0.07,
        use_fused_kernels=True,
        pooling="cls",
        freeze=False,
        trainable_logit_scale=False,
        pretrained=False,
        gradient_checkpointing=False,
        attn_implementation="flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.logit_scale = logit_scale
        self.trainable_logit_scale = trainable_logit_scale
        self.use_fused_kernels = use_fused_kernels
        self.pooling = pooling
        self.freeze = freeze
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
        self.attn_implementation = attn_implementation
