from transformers.configuration_utils import PretrainedConfig


class BiEncoderConfig(PretrainedConfig):
    model_type = "biencoder"

    def __init__(
        self,
        model_type="nomic_bert",
        model_name="EleutherAI/pythia-1b",
        projection_dim=None,
        logit_scale=1 / 0.07,
        use_fused_kernels=True,
        pooling="cls",
        freeze=False,
        trainable_logit_scale=False,
        hamming=False,
        pretrained=False,
        gradient_checkpointing=False,
        base_model_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_type = model_type
        self.base_model_config = base_model_config
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.logit_scale = logit_scale
        self.trainable_logit_scale = trainable_logit_scale
        self.use_fused_kernels = use_fused_kernels
        self.pooling = pooling
        self.freeze = freeze
        self.hamming = hamming
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
