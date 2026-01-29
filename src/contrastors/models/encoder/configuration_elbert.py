from transformers import PretrainedConfig 


class ELBertConfig(PretrainedConfig):
    model_type = "elbert"

    def __init__(
        self,
        curvature_init=None,
        curvature_max=10.0,
        curvature_min=0.1,
        trainable_curvature=None,
        projection='linear',
        pooling="euc_cls",
        hidden_size=None,
        norm_scaler=None,
        trainable_norm_scaler=None,
        norm_clip_factor=None,
        initializer_range=0.02,
        base_model_name="bert-base-uncased",
        base_model_config=None,
        delayer_norm=False,
        **kwargs,
    ):
        self.curvature_init = curvature_init
        self.curvature_max = curvature_max
        self.curvature_min = curvature_min
        self.trainable_curvature = trainable_curvature
        self.hidden_size = hidden_size
        self.norm_scaler = norm_scaler
        self.projection = projection
        self.trainable_norm_scaler = trainable_norm_scaler
        self.norm_clip_factor = norm_clip_factor
        self.base_model_name = base_model_name
        self.base_model_config = base_model_config
        self.initializer_range = initializer_range
        self.pooling = pooling
        self.delayer_norm = delayer_norm

        super().__init__(**kwargs)
