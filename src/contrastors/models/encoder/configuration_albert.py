from transformers import BertConfig


class ALBertConfig(BertConfig):
    model_type = "Albert"

    def __init__(
        self,
        num_hidden_layers=12,
        curvature_init=1.0,
        curvature_max=10.0,
        curvature_min=0.1,
        trainable_curvature=False,
        rope_theta=10000.0,
        use_rotary_embeddings=True,
        pooling="cls",
        norm_layer="layer_norm",
        attention_bias=True,
        decoder_bias=False,
        classifier_bias=True,
        classifier_activation="gelu",
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        attn_implementation="flex_attention",
        attn_heads_concat=True,
        query_key_norm=False,
        **kwargs,
    ):
        # Configuration for the hyperbolic models
        self.num_hidden_layers = num_hidden_layers
        self.curvature_init = curvature_init
        self.curvature_max = curvature_max
        self.curvature_min = curvature_min
        self.trainable_curvature = trainable_curvature
        self.rope_theta = rope_theta
        self.pooling = pooling
        self.norm_layer = norm_layer
        self.use_rotary_embeddings = use_rotary_embeddings
        self.attention_bias = attention_bias
        self.attn_heads_concat = attn_heads_concat
        self.attn_implementation = attn_implementation
        self.query_key_norm = query_key_norm

        # For masked language modeling
        self.classifier_bias = classifier_bias
        self.decoder_bias = decoder_bias
        self.classifier_activation = classifier_activation
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile

        super().__init__(**kwargs)
