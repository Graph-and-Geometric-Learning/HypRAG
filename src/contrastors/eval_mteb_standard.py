#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict
from datetime import datetime
from typing import Any
from sentence_transformers import SentenceTransformer
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
from models import HypBiEncoder, HypBiEncoderConfig
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", default="MTEB(eng, v2)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--splits", nargs="*", default=["test"])
    p.add_argument("--output_dir", default="mteb_results")
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

    safe_model = args.model.replace("/", "__")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(args.output_dir, f"{safe_model}_{run_id}")
    os.makedirs(out_folder, exist_ok=True)

    json.dump(vars(args), open(os.path.join(out_folder, "run_config.json"), "w"), indent=2)

    model = SentenceTransformer(args.model, device=args.device, trust_remote_code=True)
    model.max_seq_length = 512
    # IMPORTANT: get_benchmark returns task objects; do NOT pass the string directly as tasks
    tasks = mteb.get_benchmark(args.benchmark)
    print(f"\nRunning evaluation on {len(tasks)} tasks")
    print(f"Output folder: {out_folder}")
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        output_folder=out_folder,
        eval_splits=args.splits,
        encode_kwargs={"batch_size": args.batch_size},
    )
    out_folder = "logs/mteb_eval/albert_pretrain_oem_inner/albert_20260123_230902"
    agg = aggregate(out_folder)
    json.dump(agg, open(os.path.join(out_folder, "aggregate_summary.json"), "w"), indent=2)

    print(f"\n Done. Results in: {out_folder}")
    print(f"Mean (Task):      {agg['mean_task']}")
    print(f"Mean (TaskType):  {agg['mean_task_type']}")
    print(f"Tasks counted:    {agg['num_tasks']} | Task types: {agg['num_task_types']}")


if __name__ == "__main__":
    main()
