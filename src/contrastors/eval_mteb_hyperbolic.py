#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict
from datetime import datetime
from typing import Any

import mteb
from mteb.types import (
    Array,
    BatchedInput,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from eval.mteb_eval.mteb_task_lists import TASK_LIST_CLASSIFICATION, TASK_LIST_CLUSTERING, TASK_LIST_PAIR_CLASSIFICATION, TASK_LIST_RERANKING, TASK_LIST_RETRIEVAL, TASK_LIST_STS, TASK_LIST_SUMMARIZATION

import numpy as np
import torch
from models import HypBiEncoder, HypBiEncoderConfig, ELBertConfig, ELBertModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from mteb.models import EncoderProtocol
# Disable torch compile
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import re

def task_name(t):
    try:
        n = t.name
        if isinstance(n, str) and n:
            return n
    except Exception:
        pass

    s = str(t)
    m = re.search(r"name=['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)

    return t.__class__.__name__

def get_task_type(n):
    if n in TASK_LIST_CLASSIFICATION:
        return "classification"
    elif n in TASK_LIST_CLUSTERING:
        return "clustering"
    elif n in TASK_LIST_PAIR_CLASSIFICATION:
        return "pair_classification"
    elif n in TASK_LIST_RERANKING:
        return "reranking"
    elif n in TASK_LIST_RETRIEVAL or "HardNegatives" in n:
        return "retrieval"
    elif n in TASK_LIST_STS:
        return "sts"
    elif n in TASK_LIST_SUMMARIZATION:
        return "summarizaiton"
    else:
        return "unknown"


class BiEncoderMTEBWrapper(EncoderProtocol):
    """Wrapper to make BiEncoder compatible with MTEB's EncoderProtocol"""
    
    def __init__(self, model_path: str, hybrid: bool, scoring: str, device: str = "cuda", batch_size: int = 256):
        """
        Args:
            model_path: Path to the saved BiEncoder checkpoint
            device: Device to run inference on
            batch_size: Batch size for encoding
        """
        
        self.device = device
        self.batch_size = batch_size
        
        # Load config
        print(f"Loading BiEncoder from {model_path}")
        self.config = HypBiEncoderConfig.from_pretrained(model_path)
        if hybrid:
            # self.config.model_type = "elbert"
            self.config.model_name = model_path
            model_config = AutoConfig.from_pretrained(self.config.model_name, trust_remote_code=True)
            state_dict = torch.load(os.path.join(self.config.model_name, 'pytorch_model.bin'), map_location="cpu")
            # trim state_dict if 'trunk.' prefix exists
            if any(key.startswith("trunk.") for key in state_dict.keys()):
                state_dict = {key[len("trunk.") :]: value for key, value in state_dict.items()}
            # model_config.model_type = "elbert"
            self.model = ELBertModel.from_pretrained(
                self.config.model_name,
                add_pooling_layer=True,
                config=model_config,
            )
            self.model.load_state_dict(state_dict, strict=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "answerdotai/ModernBERT-base",
                trust_remote_code=True
            )
        else:
            self.config.model_type = "albert"
            self.config.model_name = model_path
            print(f"set config.model_type to {self.config.model_type}")
            self.model = HypBiEncoder.from_pretrained(model_path, config=self.config)
            # self.model = HypBiEncoder.load_pretrained(self.model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                trust_remote_code=True
            )
        self.model = self.model.to(device)
        self.model.eval()
        # Load tokenizer
        
        self.manifold = self.model.manifold
        # Get prefixes from config
        self.query_prefix = getattr(self.config, "query_prefix", "")
        self.document_prefix = getattr(self.config, "document_prefix", "")
        self.max_seq_length = getattr(self.config, "seq_len", 256)
        self.scoring_multiplier = 1 if scoring == "inner" else -1
        self.scoring_fn = self.manifold.pairwise_inner if scoring == "inner" else self.manifold.pairwise_geodesic_dist
        print(f"Model loaded successfully")
        print(f"Query prefix: '{self.query_prefix}'")
        print(f"Document prefix: '{self.document_prefix}'")
        print(f"Max sequence length: {self.max_seq_length}")
    
    def encode(
        self,
        inputs,
        *,
        task_metadata = None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encodes the given sentences using the BiEncoder.
        
        Args:
            inputs: DataLoader[BatchedInput] - A DataLoader that yields batches of input dicts
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The prompt type (query vs passage/document)
            **kwargs: Additional arguments
            
        Returns:
            np.ndarray of embeddings
        """
        if prompt_type == PromptType.query:
            prefix = self.query_prefix
        else:
            prefix = self.document_prefix
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in inputs:
                # Extract text from batch dict
                if 'query' in batch and prompt_type == PromptType.query:
                    batch_sentences = batch['query']
                elif 'text' in batch:
                    batch_sentences = batch['text']
                elif 'passage' in batch:
                    batch_sentences = batch['passage']
                else:
                    # Fallback: find first text-like key
                    text_keys = [k for k in batch.keys() if k not in ['id', 'metadata']]
                    if text_keys:
                        batch_sentences = batch[text_keys[0]]
                    else:
                        raise ValueError(f"Could not find text field in batch. Keys: {batch.keys()}")
                
                if prefix:
                    batch_sentences = [prefix + str(s) for s in batch_sentences]
                else:
                    batch_sentences = [str(s) for s in batch_sentences]   
        
                encoded = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
        
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Get embeddings
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        # normalize=True
                    )
                embeddings = outputs["pooler_output"]
                # all_embeddings.append(embeddings.cpu().float().numpy())
                all_embeddings.append(embeddings.float())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, 0).cpu()
        all_embeddings = torch.where(torch.isnan(all_embeddings), 0, all_embeddings)
        return all_embeddings

    def similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> Array:
            return self.scoring_multiplier * self.scoring_fn(embeddings1, embeddings2).numpy()

    def similarity_pairwise(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> Array:
        return self.scoring_multiplier * self.scoring_fn(embeddings1, embeddings2).diag().numpy()


    # @property
    # def mteb_model_meta(self):
    #     return {
    #         "name": self.config.model_name,
    #         "revision": "custom",
    #         "release_date": None,
    #     }
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to BiEncoder checkpoint or SentenceTransformer model name")
    p.add_argument("--model_type", default="biencoder", choices=["biencoder", "sentence_transformer"], 
                   help="Type of model to load")
    p.add_argument("--scoring", default="inner")
    p.add_argument("--benchmark", default="MTEB(eng, v2)")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--splits", nargs="*", default=["test"])
    p.add_argument("--output_dir", default="logs/mteb_eval")
    return p.parse_args()


def find_task_jsons(output_folder: str):
    ignore = {"run_config.json", "aggregate_summary.json"}
    out = []
    for root, _, files in os.walk(output_folder):
        for fn in files:
            if fn.endswith(".json") and fn not in ignore:
                out.append(os.path.join(root, fn))
    return sorted(out)


def pick_main_score(d: dict):
    # Try common main metrics first; fall back to any scalar found
    preferred = ["main_score", "ndcg_at_10", "map", "mrr", "accuracy", "f1", "spearman", "pearson"]
    nums = []

    def rec(x, path=""):
        if isinstance(x, (int, float)):
            nums.append((path, float(x)))
        elif isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")

    rec(d)
    for pk in preferred:
        for k, v in nums:
            if k.endswith(pk):
                return v, k
    return (nums[0][1], nums[0][0]) if nums else (None, None)


def aggregate(output_folder: str):
    per_task = []
    per_type_scores = defaultdict(list)

    for path in find_task_jsons(output_folder):
        try:
            d = json.load(open(path, "r"))
        except Exception:
            continue

        task_name = d.get("task_name") or os.path.splitext(os.path.basename(path))[0]
        task_type = get_task_type(task_name)

        score, score_key = pick_main_score(d)
        if score is None:
            continue

        per_task.append({"task": task_name, "task_type": task_type, "score": score, "score_key": score_key})
        per_type_scores[task_type].append(score)

    if not per_task:
        return {
            "num_tasks": 0,
            "num_task_types": 0,
            "mean_task": None,
            "mean_task_type": None,
            "per_task_type": {},
            "per_task": [],
            "notes": "No task JSONs with scalar scores were found. Did the benchmark run successfully?",
        }

    mean_task = sum(x["score"] for x in per_task) / len(per_task)
    per_task_type = {t: sum(v) / len(v) for t, v in per_type_scores.items()}
    mean_task_type = sum(per_task_type.values()) / len(per_task_type)

    return {
        "num_tasks": len(per_task),
        "num_task_types": len(per_task_type),
        "mean_task": mean_task,
        "mean_task_type": mean_task_type,
        "per_task_type": per_task_type,
        "per_task": per_task,
    }


def main():
    args = parse_args()

    if args.hybrid:
        safe_model = "elbert"
    else:
        safe_model = "albert"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(args.output_dir, f"{safe_model}_{run_id}")
    os.makedirs(out_folder, exist_ok=True)

    json.dump(vars(args), open(os.path.join(out_folder, "run_config.json"), "w"), indent=2)

    model = BiEncoderMTEBWrapper(
        model_path=args.model,
        hybrid=args.hybrid,
        scoring = args.scoring,
        device=args.device,
        batch_size=args.batch_size
    )
    # model = SentenceTransformerModule(model.model, model.tokenizer, model.config.seq_len, model.config.pooling)

    # Get benchmark tasks
    tasks = mteb.get_benchmark(args.benchmark)
    print(f"\nRunning evaluation on {len(tasks)} tasks")
    print(f"Output folder: {out_folder}")
    
    # Run evaluation using new API
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        output_folder=out_folder,
        eval_splits=args.splits,
        encode_kwargs={"batch_size": args.batch_size},
    )
    # Aggregate results
    agg = aggregate(out_folder)
    json.dump(agg, open(os.path.join(out_folder, "aggregate_summary.json"), "w"), indent=2)

    print(f"\n✓ Done. Results in: {out_folder}")
    print(f"Mean (Task):      {agg['mean_task']}")
    print(f"Mean (TaskType):  {agg['mean_task_type']}")
    print(f"Tasks counted:    {agg['num_tasks']} | Task types: {agg['num_task_types']}")


if __name__ == "__main__":
    main()