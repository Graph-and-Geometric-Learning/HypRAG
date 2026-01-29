import os, glob
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk, interleave_datasets
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from distributed import gather
from models import BertConfig, NomicBertForPreTraining, bert_config_to_nomic_config

from .base import BaseTrainer


def load_rank_local_iterable(shards_root: str, world_size: int, rank: int):
    shard_paths = sorted(glob.glob(os.path.join(shards_root, "shard_*")))
    my_paths = shard_paths[rank::world_size]   # each rank gets ~929/world_size shards
    ds_list = [load_from_disk(p) for p in my_paths]
    return interleave_datasets(ds_list, stopping_strategy="all_exhausted")

def split_train_val_iterable(ds, val_pct: float, seed: int):
    total = 1_857_736_518
    val_size = int(total * val_pct)

    ds_shuf = ds.shuffle(seed=seed, buffer_size=500_000)

    val = ds_shuf.take(val_size)
    train = ds_shuf.skip(val_size)
    return train, val


# TODO: add deepspeed support/check that it works and then train a mlm bert
class MLMTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(MLMTrainer, self).__init__(config, dtype)

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        with self.main_process_first():
            dataset = load_dataset(data_config.tokenized_dataset, split="train")
            tokenized_datasets = dataset.shuffle(seed=data_config.seed)
            split = tokenized_datasets.train_test_split(test_size=data_config.val_pct, seed=data_config.seed)
            train_tokenized, val_tokenized = split["train"], split["test"]

        if self.num_processes > 1:
            train_sampler = DistributedSampler(train_tokenized)
            val_sampler = DistributedSampler(val_tokenized)
        else:
            train_sampler = None
            val_sampler = None

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=data_config.mlm_prob)

        train_dataloader = DataLoader(
            train_tokenized,
            batch_size=data_config.batch_size // self.num_processes,
            shuffle=True if train_sampler is None else False,
            num_workers=data_config.workers,
            collate_fn=data_collator,
            drop_last=True,
            persistent_workers=True,
            sampler=train_sampler,
        )

        eval_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=data_config.val_mlm_prob
        )
        val_dataloader = DataLoader(
            val_tokenized,
            batch_size=data_config.batch_size // self.num_processes,
            shuffle=False,
            num_workers=data_config.workers,
            collate_fn=eval_collator,
            drop_last=True,
            persistent_workers=True,
            sampler=val_sampler,
        )

        self.total_num_steps = int(len(train_dataloader))

        return {"train": train_dataloader, "val": val_dataloader, "test": None}


    def forward_step(self, model, inputs, **kwargs):
        model.train()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model(**inputs)

        loss = output.loss
        ret = {'loss': loss}
        if hasattr(output, "info") and output.info is not None:
            ret.update(output.info)
        return ret

    def eval_step(self, model, batch, **kwargs):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        output = model(**batch, **kwargs)

        loss = output.loss

        return loss

    def eval_loop(self, model, dataloader, step, **kwargs):
        train_args = self.config.train_args
        val_loss = MeanMetric(nan_strategy="error").to(model.device)
        model.eval()
        for batch in tqdm(dataloader, desc=f"Eval epoch step {step}"):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=(not self.deepspeed and not self.config.train_args.grad_cache)):
                    loss = self.eval_step(model, batch)

            loss = gather(loss.detach().float())
            val_loss.update(loss)

        val_loss = val_loss.compute()
        ppl = torch.exp(val_loss)
        if train_args.wandb:
            self.log({"val_loss": val_loss, "val_ppl": ppl})
        else:
            self.print({"val_loss": val_loss, "val_ppl": ppl})

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_args=train_args,
            total_num_steps=total_num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        # print(f"Step {step} loss: {loss['loss'].item()}")
        # raise ValueError

        return loss
