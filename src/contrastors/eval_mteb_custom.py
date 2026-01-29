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
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

import numpy as np
import torch
from models import BiEncoder, BiEncoderConfig
from transformers import AutoTokenizer
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
    # 1) direct attribute (fast path)
    try:
        n = t.name
        if isinstance(n, str) and n:
            return n
    except Exception:
        pass

    # 2) parse from repr/str:  ArguAna(name='ArguAna', languages=['eng'])
    s = str(t)
    m = re.search(r"name=['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)

    # 3) fallback: class name
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
    
    def __init__(self, model_path: str, base_model_type: str, device: str = "cuda", batch_size: int = 256):
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
        self.model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
        self.config = BiEncoderConfig.from_pretrained(model_path)
        # self.config.base_model_type = base_model_type
        # # Load model
        # self.model = BiEncoder.from_pretrained(model_path, config=self.config)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            trust_remote_code=True
        )
        
        # Get prefixes from config
        self.query_prefix = getattr(self.config, "query_prefix", "")
        self.document_prefix = getattr(self.config, "document_prefix", "")
        self.max_seq_length = getattr(self.config, "seq_len", 128)
        
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
                embeddings = outputs["last_hidden_state"].mean(1)
                if len(embeddings) == 0:
                    print(f"Warning: Empty embeddings for sentences: {batch_sentences[:3]}...")
                all_embeddings.append(embeddings.cpu().float().numpy())

        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        return embeddings1@embeddings2.T + 1e-4
    
    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        return (embeddings1 * embeddings2).sum(1) + 1e-4


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
    p.add_argument("--benchmark", default="MTEB(eng, v2)")
    p.add_argument("--base_model_type", default="modernbert")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
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

    for path in find_task_jsons(output_folder)[:-1]:
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

    safe_model = args.base_model_type
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(args.output_dir, f"{safe_model}_{run_id}")

    os.makedirs(out_folder, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_folder, "run_config.json"), "w"), indent=2)

    model = BiEncoderMTEBWrapper(
        model_path=args.model,
        base_model_type=args.base_model_type,
        device=args.device,
        batch_size=args.batch_size
    )
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
        ignore_metrics=["naucs"],
        compute_abstention=False
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