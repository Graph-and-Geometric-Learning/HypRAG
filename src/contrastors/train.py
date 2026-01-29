import logging
import os
import fire
from argparse import ArgumentParser, ArgumentTypeError
from dotenv import load_dotenv

import deepspeed
import torch
import torch.distributed as dist


from read import read_config
from trainers import TRAINER_REGISTRY

import datetime

logger = logging.getLogger('s3fs')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a console handler and set its level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the console handler
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)

DTYPE_MAPPING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def update_config(config, kwargs):
    for key, value in kwargs.items():
        key_parts = key.split(".")
        cur_attr = config
        for part in key_parts[:-1]:
            if not hasattr(cur_attr, part):
                raise ValueError(f"Key {key} does not exist in config")
            cur_attr = getattr(cur_attr, part)

        if not hasattr(cur_attr, key_parts[-1]):
            raise ValueError(f"Key {key} does not exist in config")
        setattr(cur_attr, key_parts[-1], value)

    return config


def main(**kwargs):
    config_path = kwargs.pop("config", None)
    deepspeed_config = kwargs.pop("deepspeed_config", None)

    config = read_config(config_path)

    config = update_config(config, kwargs)
    # allowed_keys = set(config.__dict__.keys())  # or use dataclasses.fields if it's a dataclass
    # print(f"[INFO] Allowed keys in config: {allowed_keys}")
    # for key, value in kwargs.items():
    #     if key in allowed_keys:
    #         setattr(config, key, value)
    #     else:
    #         print(f"[INFO] Skipping CLI arg '{key}' not in config schema.")

    if deepspeed_config is not None:
        config.deepspeed = True
        config.deepspeed_config = deepspeed_config

        # to load deepspeed checkpoint with weights_only=True
        from torch.serialization import add_safe_globals
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        add_safe_globals([LossScaler])

    if deepspeed_config is not None:
        deepspeed.init_distributed()
    else:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if config.train_args.output_dir is not None:
        config.train_args.wandb_run_name = config.train_args.output_dir.split("/")[-1]


    trainer_cls = TRAINER_REGISTRY[config.train_args.trainer_type]

    dtype = DTYPE_MAPPING[config.train_args.dtype]

    # torch.cuda.memory._record_memory_history(max_entries=100000)

    trainer = trainer_cls(config, dtype)
    trainer.train()

    # torch.cuda.memory._dump_snapshot("memory_snapshot.json")
    # torch.cuda.memory._record_memory_history(enabled=None)

    dist.destroy_process_group()


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    # set print options suppress scientific notation for torch and np
    torch.set_printoptions(sci_mode=False)
    # torch.autograd.set_detect_anomaly(True)

    fire.Fire(main)
