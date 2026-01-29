from .biencoder import *
from .decoder import *
from .encoder import *
from .model_utils import *
from .vit import *


MODEL_REGISTRY = {
    **ENCODER_REGISTRY,
}

MODEL_LOADER_REGISTRY = {
    **ENCODER_LOADER_REGISTRY,
}


def load_from_config(config):
    model_loader = MODEL_LOADER_REGISTRY.get(config.model_type)
    print("Inside model init load from config")
    model = model_loader(config)

    return model
