import torch.distributed
import yaml
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import DataLoader
from torch import Tensor
from numpy import ndarray
import hyplib.nn as hnn
from hyplib.manifolds import Lorentz

from dataset.text_text_loader import StreamingShardDataset, collate_fn, get_local_dataloader
from distributed import gather_with_grad, print_in_order
from hyperbolic_loss import clip_loss, grad_cache_loss
from models import HypBiEncoder, HypBiEncoderConfig, LogitScale, load_tokenizer
from megablocks.layers import moe
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.util import cos_sim

from .base import BaseTrainer

from typing import List, Dict, Union, Tuple, Callable


def convert_distance_to_similarity_function(distance_fn):
    def similarity_function(query: Tensor, document: Tensor) -> Tensor:
        return -distance_fn(query, document)
    return similarity_function


class HypSentenceTransformerModule(nn.Module):
    """
    Wrapper class to make a custom embedding model compatible with SentenceTransformers.

    Args:
        custom_model: Your custom PyTorch module that generates embeddings
        model_name_or_path: Name/path of the tokenizer to use (e.g., 'bert-base-uncased')
        max_seq_length: Maximum sequence length for the tokenizer
        do_lower_case: Whether to lowercase input text
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_seq_length: int = 128,
        pooling: str = None,
        distance: str = "lorentz_inner",
    ):
        # Initialize as nn.Sequential with your custom model as the only module
        super().__init__()
        self.model = model
        
        # Store configuration
        self.max_seq_length = max_seq_length
        self.do_lower_case = False
        
        # Initialize tokenizer
        self.tokenizer = tokenizer

        self.pooling = pooling
        self.distance = distance

    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass that matches SentenceTransformers expected format"""
        # Get embeddings from your custom model
        keep_keys = ["input_ids", "attention_mask", "token_type_ids"]
        features = {k: features[k] for k in keep_keys if k in features}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                embeddings = self.model(**features)
        if 'pooler_output' in embeddings.keys():
            return {"sentence_embedding": embeddings["pooler_output"]}
        else:
            return {"sentence_embedding": embeddings["embedding"]}

    def tokenize(
        self,
        texts: Union[List[str], List[Dict], List[Tuple[str, str]]],
        padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes text input in the format expected by SentenceTransformers"""
        output = {}

        # Handle different input formats
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # Preprocess text
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Perform tokenization
        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length
            )
        )
        return output


class HypSentenceTransformer(SentenceTransformer):
    def __init__(self, modules: nn.Module, distance='lorentz_inner', *args, **kwargs):
        super().__init__(modules=modules, *args, **kwargs)
        self.distance = distance
        manifold = modules[0].model.manifold

        if distance == 'lorentz_inner':
            self.similarity_fn = manifold.l_inner
            self.similarity_pairwise_fn = manifold.pairwise_inner
        elif distance == 'geodesic':
            self.similarity_fn = convert_distance_to_similarity_function(manifold.induced_distance)
            self.similarity_pairwise_fn = convert_distance_to_similarity_function(manifold.pairwise_geodesic_dist)
        elif distance == 'squared_lorentz':
            self.similarity_fn = convert_distance_to_similarity_function(manifold.sqdist)
            self.similarity_pairwise_fn = convert_distance_to_similarity_function(manifold.pairwise_squared_dist)
        elif distance == 'euclidean':
            self.similarity_fn = cos_sim
            self.similarity_pairwise_fn = cos_sim
        else:
            raise ValueError(f"Distance {distance} not supported")

    def similarity(self, query: Tensor, document: Tensor) -> Tensor:
        return self.similarity_pairwise_fn(query, document)

    def similarity_pairwise(self, query: Tensor, document: Tensor) -> Tensor:
        return self.similarity_pairwise_fn(query, document)


class HypTextTextTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(HypTextTextTrainer, self).__init__(config, dtype)
        self.use_grad_cache = config.train_args.grad_cache
        self.matryoshka_dims = config.train_args.matryoshka_dims
        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                config.train_args.matryoshka_loss_weights
                if config.train_args.matryoshka_dims and config.train_args.matryoshka_loss_weights
                else [1] * len(config.train_args.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

    def get_model(self, config):
        model_config = config.model_args

        if model_config.checkpoint is None:
            model_config_dict = model_config.dict()
            config = HypBiEncoderConfig(**model_config_dict)
            model = HypBiEncoder(config)
        else:
            self.print(f"Loading model from {model_config.checkpoint}")
            loaded_config = HypBiEncoderConfig.from_pretrained(model_config.checkpoint)
            if model_config.projection_dim is not None:
                loaded_config.projection_dim = model_config.projection_dim
            if model_config.gradient_checkpointing:
                loaded_config.gradient_checkpointing = True
            model = HypBiEncoder.from_pretrained(model_config.checkpoint, config=loaded_config)
            config = loaded_config

        if self.distributed and not self.deepspeed:
            model = model.to("cuda")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.process_index],
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        scale = LogitScale(config)

        if self.distributed and not self.deepspeed:
            scale = scale.to("cuda")
            if sum(p.requires_grad for p in scale.parameters()) > 0:
                scale = torch.nn.parallel.DistributedDataParallel(
                    scale,
                    device_ids=[self.process_index],
                )

        tokenizer = load_tokenizer(model_config)

        return {"model": model, "logit_scale": scale, "config": config, "tokenizer": tokenizer}

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        model_args = config.model_args
        gradient_accumulation_steps = train_args.gradient_accumulation_steps
        if train_args.wandb_run_name is None and train_args.wandb:
            raise ValueError("wandb_run_name must be set, got None")

        if data_config.batch_size % self.num_processes != 0:
            raise ValueError(
                f"Batch size {data_config.batch_size} must be divisible by accelerator.num_processes {self.num_processes}"
            )

        batch_size = int(data_config.batch_size / self.num_processes)
        train_dataloader = get_local_dataloader(
            data_config.input_shards,
            batch_size,
            self.tokenizer,
            seed=data_config.seed,
            streaming=data_config.streaming,
            num_negatives=model_args.num_negatives,
            query_max_length=data_config.query_max_length,
            document_max_length=data_config.document_max_length,
            add_prefix=model_args.add_prefix,
            num_workers=data_config.workers,
            epoch=0,
            data_path=data_config.data_path,
            shuffle_buffer_size=data_config.shuffle_buffer_size,
        )
        self.total_num_steps = int(
            len(train_dataloader.dataset) / gradient_accumulation_steps // data_config.batch_size
        ) if train_args.num_train_steps is None else train_args.num_train_steps

        nano_beir = NanoBEIREvaluator(query_prompts=model_args.query_prefix, corpus_prompts=model_args.document_prefix, show_progress_bar=True)

        return {"train": train_dataloader, "val": nano_beir, "test": None}

    def save_model(self, output_dir):
        super().save_model(output_dir)
        if self.global_rank == 0:
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

                
    def load_model(self, model_path):
        config = HypBiEncoderConfig.from_pretrained(model_path)
        loaded_model = HypBiEncoder.from_pretrained(model_path, config=config)
        loaded_model = loaded_model.to("cuda")
        if isinstance(self.model["model"],(nn.parallel.DistributedDataParallel, nn.DataParallel, deepspeed.DeepSpeedEngine)):
            torch.distributed.barrier()
            loaded_model = torch.nn.parallel.DistributedDataParallel(
                loaded_model,
                device_ids=[self.process_index],
                # find_unused_parameters=True,
                broadcast_buffers=False,
            )

        return loaded_model

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def forward_step(self, model, inputs, logit_scale, **kwargs):
        model.train()
        if self.use_grad_cache:
            loss = self._grad_cache_forward_step(model, inputs, logit_scale, **kwargs)
        else:
            loss = self._forward_step(
                model=model,
                batch=inputs,
                logit_scale=logit_scale,
                matryoshka_dims=self.matryoshka_dims,
                matroyshka_loss_weights=self.matryoshka_loss_weights,
                **kwargs,
            )

        return loss

    def backward(self, loss):
        if isinstance(loss, dict):
            loss = loss["loss"]

        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            # grad cache backprops in the loss function, becomes a noop
            if not self.use_grad_cache:
                loss.backward()

    def _grad_cache_forward_step(self, model, batch, logit_scale, **kwargs):
        # TODO: could pass this to grad cache loss and log?
        if 'dataset_name' in batch:
            batch.pop("dataset_name")
        step = kwargs.pop("step")
        batch = {k: v.to(model.device) for k, v in batch.items()}

        query_inputs = {k.replace("query_", ""): v for k, v in batch.items() if "query" in k}
        document_inputs = {k.replace("document_", ""): v for k, v in batch.items() if "document" in k}

        loss = grad_cache_loss(
            tower1=model,
            tower2=model,
            t1_inputs=query_inputs,
            t2_inputs=document_inputs,
            chunk_size=self.config.train_args.chunk_size,
            logit_scale=logit_scale,
            distance=self.config.train_args.distance,
            tracker=self.tracker,
            step=step,
            dtype=self.dtype,
            **kwargs,
        )
        return {"loss": loss}

    def _forward_step(self, model, batch, logit_scale, matryoshka_dims=None, matroyshka_loss_weights=None, **kwargs):
        dataset_name = batch.pop("dataset_name", "noname")
        kwargs.pop("config")
        kwargs.pop("tokenizer")

        max_length = batch["document_input_ids"].shape[1] 
        padded_query_inputs = {
                "input_ids": batch["query_input_ids"].to(model.device), 
                "attention_mask": batch["query_attention_mask"].to(model.device),
        }

        query_outputs = model(
            **padded_query_inputs,
        )
        document_outputs = model(
            input_ids=batch["document_input_ids"].to(model.device),
            attention_mask=batch["document_attention_mask"].to(model.device),
        )

        if "negative_input_ids" in batch:
            raise NotImplementedError("Negative sampling not supported for text-text models")

        queries = query_outputs["embedding"]
        all_documents = gather_with_grad(document_outputs["embedding"])
        manifold = query_outputs["manifold"]

        if matryoshka_dims:
            loss = 0.0
            for loss_weight, dim in zip(matroyshka_loss_weights, matryoshka_dims):
                reduced_q = queries[:, :dim]
                reduced_d = all_documents[:, :dim]

                name_with_dim = f"{dataset_name}_matryoshka_{dim}"

                dim_loss = clip_loss(
                    query=reduced_q,
                    document=reduced_d,
                    logit_scale=logit_scale,
                    tracker=self.tracker,
                    dataset=name_with_dim,
                    manifold=manifold,
                    distance=self.config.train_args.distance,
                    dtype=self.dtype,
                    **kwargs,
                )

                loss += loss_weight * dim_loss
        else:
            loss = clip_loss(
                query=queries,
                document=all_documents,
                logit_scale=logit_scale,
                tracker=self.tracker,
                dataset=dataset_name,
                manifold=manifold,
                distance=self.config.train_args.distance,
                dtype=self.dtype,
                **kwargs,
            )

        return {"loss": loss}

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

        if train_args.clamp_logits:
            with torch.no_grad():
                self.model["scale"].module.logit_scale.clamp_(0, np.log(train_args.logit_max))

        if train_args.wandb:
            if isinstance(loss, dict):
                self.log({k: v.detach().cpu().item() for k, v in loss.items()}, step=step)

        return loss

    def eval_loop(self, model, dataloader, step, **kwargs):
        model.eval()
        train_args = self.config.train_args
        model_args = self.config.model_args
        if self.process_index == 0:
            original_model = model.module
            module = nn.Sequential(HypSentenceTransformerModule(
                model=original_model,
                max_seq_length=model_args.seq_len,
                tokenizer=self.tokenizer,
                pooling=model_args.pooling,
                distance=train_args.distance,
            ))
            emb = HypSentenceTransformer(modules=module, distance=train_args.distance)
            results = dataloader(emb)

            ndcg = {f'beir/{k.replace("Nano", "").replace("_cosine", "").lower()}': v for k, v in results.items() if "ndcg@10" in k}

            if train_args.wandb:
                self.log(ndcg, step=step)

            self.print(ndcg)
        torch.distributed.barrier()
