# save as hf_to_mds.py
import json
import os
import argparse
from typing import Iterable, Dict, Any, List, Optional
from datasets import load_dataset, load_dataset_builder
from streaming import MDSWriter
from tqdm import tqdm

ALL_SPLITS = [
    "reddit_title_body", "amazon_reviews", "paq",
    "s2orc_citation_titles", "s2orc_title_abstract",
    "s2orc_abstract_citation", "s2orc_abstract_body",
    "wikianswers", "wikipedia", "gooaq", "codesearch",
    "yahoo_title_answer", "agnews", "amazonqa", "yahoo_qa",
    "yahoo_title_question", "ccnews", "npr", "eli5", "cnn",
    "stackexchange_duplicate_questions", "stackexchange_title_body",
    "stackexchange_body_body", "sentence_compression", "wikihow",
    "altlex", "quora", "simplewiki", "squad",
]

COLUMNS = {
    # dataset schema:
    # query: str, document: str, dataset: str, shard: int64
    "query": "str",
    "document": "str",
    "dataset": "str",
    "shard": "int",
}

def iter_hf_split(split: str, streaming: bool = True, hf_token: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """
    Create an iterator over a single HF split. Uses streaming to avoid local caching.
    """
    ds = load_dataset(
        "nomic-ai/nomic-embed-unsupervised-data",
        split=split,
        streaming=streaming,
        token=hf_token,
    )
    for ex in ds:
        # Ensure types are clean for MDS encoders
        yield {
            "query": "" if ex["query"] is None else str(ex["query"]),
            "document": "" if ex["document"] is None else str(ex["document"]),
            "dataset": "" if ex["dataset"] is None else str(ex["dataset"]),
            "shard": int(ex["shard"]) if ex["shard"] is not None else 0,
        }

def write_split_to_mds(
    split: str,
    out_root: str,
    size_limit: str = "128mb",
    compression: str = "zstd",
    hashes: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    show_pbar: bool = True,
    hf_token: Optional[str] = None,
):
    """
    Stream a split from HF and write to local MDS shards.
    Output directory: {out_root}/{split}
    """
    out_dir = f"{out_root.rstrip('/')}/{split}"
    rows = iter_hf_split(split, streaming=True, hf_token=hf_token)

    with MDSWriter(
        out=out_dir,
        columns=COLUMNS,
        compression=compression,
        size_limit=size_limit,
        hashes=hashes or ["sha1"],
        progress_bar=False,  # we wrap with tqdm
    ) as writer:
        it = rows
        if show_pbar:
            it = tqdm(it, desc=f"Writing {split}", unit="rows", smoothing=0.01)
        for i, row in enumerate(it, 1):
            writer.write(row)
            if max_rows is not None and i >= max_rows:
                break

def main():
    ap = argparse.ArgumentParser(description="Convert HF dataset to local Mosaic MDS.")
    ap.add_argument("--out", required=True, help="Output root directory for MDS (created if missing).")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        help=f"Which splits to convert (default: all {len(ALL_SPLITS)} splits).",
    )
    ap.add_argument("--size-limit", default="128mb", help='Target shard size (e.g., "64mb", "200mb").')
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"], help="Shard compression.")
    ap.add_argument("--hash", dest="hashes", nargs="*", default=["sha1"], help="Per-shard hash(es).")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows per split (for smoke tests).")
    ap.add_argument("--no-pbar", action="store_true", help="Disable progress bars.")
    ap.add_argument("--hf-token", default=None, help="Hugging Face token if required in your env.")
    args = ap.parse_args()

    for split in args.splits:
        if split not in ALL_SPLITS:
            raise ValueError(f"Unknown split: {split}. Valid: {ALL_SPLITS}")
        write_split_to_mds(
            split=split,
            out_root=args.out,
            size_limit=args.size_limit,
            compression=args.compression,
            hashes=args.hashes,
            max_rows=args.max_rows,
            show_pbar=not args.no_pbar,
            hf_token=args.hf_token,
        )

    # write dataset metadata
    meta_out = os.path.join(args.out, "metadata.json")
    with open(meta_out, "w") as f:
        dataset_builder = load_dataset_builder("nomic-ai/nomic-embed-unsupervised-data", use_auth_token=args.hf_token)
        split_infos = {
            split: {
                "num_bytes": dataset_builder.info.splits[split].num_bytes,
                "num_examples": dataset_builder.info.splits[split].num_examples,
            }
            for split in args.splits if split in dataset_builder.info.splits
        }
        dict_meta = {
            "source": "nomic-ai/nomic-embed-unsupervised-data",
            "splits": args.splits,
            "columns": COLUMNS,
            "size_limit": args.size_limit,
            "compression": args.compression,
            "hashes": args.hashes,
            "split_infos": split_infos,
        }
        json.dump(dict_meta, f, indent=2)
        


if __name__ == "__main__":
    main()
