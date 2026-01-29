import os
import json
import argparse
from typing import Optional, List, Any
from datetime import datetime
import numpy as np
import faiss

import torch
import torch.nn as nn

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ContextRelevance,
    AnswerRelevancy,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from langchain_core.embeddings import Embeddings as BaseLangchainEmbeddings
from langchain_community.llms import HuggingFacePipeline

from models import BiEncoder, BiEncoderConfig, HypBiEncoder, HypBiEncoderConfig, ELBertModel
from trainers.text_text import SentenceTransformerModule
from trainers.hyp_text_text import HypSentenceTransformerModule, HypSentenceTransformer

# -------------------------
# Args / Env
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="BiEncoder model path")
    parser.add_argument("--base_model_type", default="modernbert")
    parser.add_argument("--hyperbolic", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--hf", action="store_true")

    parser.add_argument("--dataset", required=True, help="RAGBench config name (e.g., delucionqa)")
    parser.add_argument("--split", default="test", help="HF split (default: test)")
    parser.add_argument("--output", required=True, help="Output directory")

    # Qwen3 evaluator LLM
    parser.add_argument("--qwen_model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model id for Qwen3")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)

    # Embeddings
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    
    # Retrieval
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--hyp_distance", default="squared_lorentz", choices=["lorentz", "squared_lorentz", "geodesic"],
                       help="Hyperbolic distance type")

    return parser.parse_args()

def validate_env():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN must be set to download dataset (and possibly HF models).")



class HFSentenceTransformerEmbeddings(BaseLangchainEmbeddings):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 256,
        normalize: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.st_model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True
        )
        self.model = model_name
        self.normalize = normalize

    def embed_documents(self, texts):
        vecs = self.st_model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, query):
        vec = self.st_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

class EucSentenceTransformerEmbeddings(BaseLangchainEmbeddings):
    """
    Adapter for your BiEncoder -> SentenceTransformer -> LangChain Embeddings interface.
    """
    def __init__(self, model_path: str, base_model_type: str, device: str = "cuda", batch_size: int = 256):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.encoder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.encoder = self.encoder.to(device)
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.base_model_type = base_model_type
        self.encoder.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            trust_remote_code=True
        )
        self.max_seq_length = self.config.seq_len if hasattr(self.config, "seq_len") else 256
        self.model = str(model_path)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Embed a single batch of texts"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get embeddings
        with torch.no_grad():  # Important: disable gradients for inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        
        # Mean pooling over sequence dimension
        embeddings = outputs["last_hidden_state"].mean(dim=1)
        return embeddings.cpu()  # Move to CPU to free GPU memory

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in batches to avoid OOM"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Clear cache periodically
            if (i // self.batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.float().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        vec = self.embed_batch([query])[0]
        return vec.float().numpy().tolist()

class HypSentenceTransformerEmbeddings(BaseLangchainEmbeddings):
    """
    Adapter for your BiEncoder -> SentenceTransformer -> LangChain Embeddings interface.
    """
    def __init__(self, model_path: str, base_model_type: str, hybrid: bool, scoring: str, device: str = "cuda", batch_size: int = 256):
        super().__init__()
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
            self.encoder = ELBertModel.from_pretrained(
                self.config.model_name,
                add_pooling_layer=True,
                config=model_config,
            )
            self.encoder.load_state_dict(state_dict, strict=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "answerdotai/ModernBERT-base",
                trust_remote_code=True
            )
        else:
            self.config.model_type = "albert"
            self.config.model_name = model_path
            # # Load model
            self.encoder = HypBiEncoder.from_pretrained(model_path, config=self.config)
            self.encoder = HypBiEncoder.load_pretrained(self.encoder, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                trust_remote_code=True
            )
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        # Load tokenizer
        self.model = str(model_path)
        module = nn.Sequential(HypSentenceTransformerModule(
                model=self.encoder,
                max_seq_length=256,
                tokenizer=self.tokenizer,
                pooling=256,
                distance="lorentz_inner",
            ))
        self.emb = HypSentenceTransformer(modules=module, distance="geodesic")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.emb.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        # vecs = torch.where(torch.isnan(vecs), 0, vecs)

        return vecs.tolist()

    def embed_query(self, query: str) -> List[float]:
        vec = self.emb.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        # vec = torch.where(torch.isnan(vec), 0, vec)
        return vec.tolist()
    
def get_ragas_embeddings(hf:bool, hyperbolic:bool, hybrid:bool, model_path: str, base_model_type: str, device: str = "cuda", batch_size: int = 256):
    if hf:
        custom_emb = HFSentenceTransformerEmbeddings(
        model_name=model_path,
        device=device,
        batch_size=batch_size,
    )
    else:
        if not hyperbolic:
            custom_emb = EucSentenceTransformerEmbeddings(model_path, base_model_type, device, batch_size)
        else:
            custom_emb = HypSentenceTransformerEmbeddings(model_path, base_model_type, hybrid, "lorentz_inner", device, batch_size)
    return LangchainEmbeddingsWrapper(custom_emb)


# -------------------------
# Load dataset
# -------------------------
def load_ragbench_split(name: str, split: str = "test") -> Dataset:
    return load_dataset("galileo-ai/ragbench", name, split=split)


def convert_to_ragas_eval(hf_ds: Dataset) -> EvaluationDataset:
    """
    RAGAS expects (for these metrics):
      - user_input: str
      - response: str
      - retrieved_contexts: List[str]
    """
    ds = hf_ds

    # map question -> user_input
    if "question" in ds.column_names:
        ds = ds.rename_column("question", "user_input")
    elif "query" in ds.column_names:
        ds = ds.rename_column("query", "user_input")
    else:
        raise ValueError(f"Can't find question/query column. Available: {ds.column_names}")

    # response stays response (or rename if needed)
    if "response" not in ds.column_names:
        if "answer" in ds.column_names:
            ds = ds.rename_column("answer", "response")
        else:
            raise ValueError(f"Can't find response/answer column. Available: {ds.column_names}")

    # documents -> retrieved_contexts
    if "documents" in ds.column_names:
        ds = ds.rename_column("documents", "retrieved_contexts")
    elif "context" in ds.column_names:
        ds = ds.rename_column("context", "retrieved_contexts")
    else:
        raise ValueError(f"Can't find documents/context column. Available: {ds.column_names}")

    # Ensure retrieved_contexts is List[str]
    def normalize_contexts(ex):
        ctx = ex["retrieved_contexts"]
        # sometimes a single string
        if isinstance(ctx, str):
            ex["retrieved_contexts"] = [ctx]
            return ex
        # sometimes list of dicts
        if isinstance(ctx, list) and len(ctx) > 0 and isinstance(ctx[0], dict):
            ex["retrieved_contexts"] = [d.get("text", "") for d in ctx]
            return ex
        return ex

    ds = ds.map(normalize_contexts)

    # Keep only what we need (optional but avoids surprises)
    keep_cols = ["user_input", "response", "retrieved_contexts"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    return EvaluationDataset.from_hf_dataset(ds)

# -------------------------
# Run evaluation
# -------------------------
def run_evaluation(
    ragas_ds: EvaluationDataset,
    embeddings_model,
    evaluator_llm,
    metrics,
    output_path: str,
):
    run_config = RunConfig(
        max_workers=1,          # Keep low for local models
        timeout=1200,           # Increase timeout to 20 minutes
        max_retries=5,          # Increase retries from 1 to 5
        max_wait=180,           # Max wait between retries
        exception_types=(Exception,),  # Catch all exceptions
    )

    results = evaluate(
        dataset=ragas_ds,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings_model,
        run_config=run_config,
        raise_exceptions=False,  # Don't stop on individual failures
    )

    scores_df = results.to_pandas()

    os.makedirs(output_path, exist_ok=True)
    scores_path = os.path.join(output_path, "ragas_scores.jsonl")
    with open(scores_path, "w") as f:
        for _, row in scores_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    summary = scores_df.mean(numeric_only=True).to_dict()
    summary_path = os.path.join(output_path, "ragas_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved annotated results to:", scores_path)
    print("Saved summary metrics to:", summary_path)

    return scores_df

def lorentz_inner_product(x, y):
    """
    Lorentz inner product: <x, y>_L = -x[0]*y[0] + x[1:] · y[1:]
    x, y: shape (..., dim+1) where first coordinate is time
    """
    return -x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)

def lorentz_distance(x, y):
    """
    Lorentz distance: d_L(x,y) = arcosh(-<x,y>_L)
    """
    inner = -lorentz_inner_product(x, y)
    # Clamp to avoid numerical issues
    inner = np.clip(inner, 1.0, None)
    return np.arccosh(inner)

def squared_lorentz_distance(x, y):
    """
    Squared Lorentz distance (often used for efficiency)
    """
    return lorentz_distance(x, y) ** 2


class HyperbolicRetriever:
    """Retriever using hyperbolic distance metrics"""
    def __init__(self, embeddings, documents: List[str], top_k: int = 5, distance_type: str = "lorentz"):
        self.embeddings = embeddings  # Raw embeddings object
        self.documents = documents
        self.top_k = top_k
        self.distance_type = distance_type
        
        print(f"Building hyperbolic index for {len(documents)} documents...")
        # Use embed_documents method directly
        doc_embeddings = self.embeddings.embed_documents(documents)
        self.doc_embeddings = np.array(doc_embeddings).astype('float32')
        print(f"Document embeddings shape: {self.doc_embeddings.shape}")
        print("Hyperbolic index built!")
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve top-k documents for a query using hyperbolic distance"""
        query_emb = np.array([self.embeddings.embed_query(query)]).astype('float32')
        
        # Compute hyperbolic distances
        if self.distance_type == "lorentz":
            # Use negative Lorentz inner product as similarity (higher is more similar)
            similarities = -lorentz_inner_product(query_emb, self.doc_embeddings.T)
            top_indices = np.argsort(similarities[0])[:self.top_k]
        elif self.distance_type == "squared_lorentz":
            distances = np.array([squared_lorentz_distance(query_emb[0], doc_emb) 
                                 for doc_emb in self.doc_embeddings])
            top_indices = np.argsort(distances)[:self.top_k]
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return [self.documents[idx] for idx in top_indices]


class EuclideanRetriever:
    """FAISS-based retriever for Euclidean embeddings"""
    def __init__(self, embeddings, documents: List[str], top_k: int = 5):
        self.embeddings = embeddings  # Raw embeddings object
        self.documents = documents
        self.top_k = top_k
        
        print(f"Building Euclidean index for {len(documents)} documents...")
        doc_embeddings = self.embeddings.embed_documents(documents)
        doc_embeddings = np.array(doc_embeddings).astype('float32')
        
        # Create FAISS index
        dim = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(doc_embeddings)
        self.index.add(doc_embeddings)
        print("Euclidean index built!")
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve top-k documents for a query"""
        query_emb = np.array([self.embeddings.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, self.top_k)
        return [self.documents[idx] for idx in indices[0]]

# -------------------------
# Factory function to create appropriate retriever
# -------------------------
def create_retriever(embeddings, documents: List[str], top_k: int = 5, 
                     is_hyperbolic: bool = False, distance_type: str = "lorentz"):
    if is_hyperbolic:
        return HyperbolicRetriever(embeddings, documents, top_k, distance_type)
    else:
        return EuclideanRetriever(embeddings, documents, top_k)


def build_document_corpus(hf_ds: Dataset) -> List[str]:
    """Extract all unique documents from the dataset"""
    all_docs = set()
    
    # Find the documents column
    doc_col = None
    for col in ["documents", "context", "retrieved_contexts"]:
        if col in hf_ds.column_names:
            doc_col = col
            break
    
    if not doc_col:
        raise ValueError(f"No documents column found in {hf_ds.column_names}")
    
    print(f"Extracting documents from '{doc_col}' column...")
    for row in hf_ds:
        docs = row[doc_col]
        
        # Handle different formats
        if isinstance(docs, str):
            all_docs.add(docs)
        elif isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, str):
                    all_docs.add(doc)
                elif isinstance(doc, dict):
                    all_docs.add(doc.get("text", ""))
    
    corpus = list(all_docs)
    print(f"Built corpus with {len(corpus)} unique documents")
    return corpus


# -------------------------
# Build LangChain Qwen3 evaluator LLM
# -------------------------
def build_qwen3_ragas_llm(model_id, max_new_tokens=512, temperature=0.6):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Create a custom pipeline with better sampling parameters
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,  # Use non-zero temperature
        top_p=0.9,  # Slightly lower for more focused outputs
        top_k=50,
        do_sample=True,
        return_full_text=False,
        repetition_penalty=1.0,
    )
    
    # Modify tokenizer's chat template to disable thinking
    original_apply_chat_template = tokenizer.apply_chat_template
    def patched_apply_chat_template(*args, **kwargs):
        kwargs['enable_thinking'] = False
        return original_apply_chat_template(*args, **kwargs)
    tokenizer.apply_chat_template = patched_apply_chat_template
    
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    
    return LangchainLLMWrapper(hf_llm), pipe

# -------------------------
# Generate Responses
# -------------------------
def generate_response_with_context(query: str, contexts: List[str], generation_pipeline) -> str:
    """Generate a response given query and retrieved contexts"""
    # Format contexts
    context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query}

Answer:"""
    
    # Generate response using the raw pipeline
    messages = [{"role": "user", "content": prompt}]
    response = generation_pipeline(messages)[0]['generated_text']
    
    return response


# -------------------------
# Create RAG Dataset
# -------------------------
def create_rag_dataset(
    hf_ds: Dataset,
    retriever,
    generation_pipeline,  # Raw pipeline, not wrapper
) -> EvaluationDataset:
    """
    For each query:
    1. Retrieve documents using embedding model
    2. Generate response using LLM
    3. Create RAGAS dataset with generated responses
    """
    rag_data = []
    
    # Find query column
    query_col = "question" if "question" in hf_ds.column_names else "query"
    
    print(f"Generating responses for {len(hf_ds)} queries...")
    for i, row in enumerate(hf_ds):
        if i % 10 == 0:
            print(f"Processing {i}/{len(hf_ds)}...")
        
        query = row[query_col]
        
        # Retrieve documents
        retrieved_contexts = retriever.retrieve(query)
        
        # Generate response
        response = generate_response_with_context(query, retrieved_contexts, generation_pipeline)
        
        rag_data.append({
            "user_input": query,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        })
    
    # Convert to HF Dataset then to RAGAS
    rag_hf_ds = Dataset.from_list(rag_data)
    return EvaluationDataset.from_hf_dataset(rag_hf_ds)


def get_retrieval_embeddings(hf:bool, hyperbolic:bool, hybrid:bool, model_path: str, base_model_type: str, device: str = "cuda", batch_size: int = 256):
    """Get embeddings for document retrieval (your custom model)"""
    if hf:
        custom_emb = HFSentenceTransformerEmbeddings(
            model_name=model_path,
            device=device,
            batch_size=batch_size,
        )
    else:
        if not hyperbolic:
            custom_emb = EucSentenceTransformerEmbeddings(model_path, base_model_type, device, batch_size)
        else:
            custom_emb = HypSentenceTransformerEmbeddings(model_path, base_model_type, hybrid, "lorentz_inner", device, batch_size)
    return custom_emb  # Return raw embeddings, not wrapped


def get_ragas_embeddings(device: str = "cuda", batch_size: int = 256):
    """Get embeddings for RAGAS evaluation metrics (always Qwen)"""
    qwen_emb = HFSentenceTransformerEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device=device,
        batch_size=batch_size,
    )
    return LangchainEmbeddingsWrapper(qwen_emb)

if __name__ == "__main__":
    args = parse_args()
    args.top_k = int(args.top_k)
    validate_env()

    safe_model = args.base_model_type
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(args.output, f"{safe_model}_{args.dataset}_{run_id}")

    print(f"Loading RAGBench split: {args.dataset} / {args.split}")
    raw_ds = load_ragbench_split(args.dataset, split=args.split)

    # Get retrieval embeddings (your custom model)
    print("Initializing custom retrieval embeddings...")
    retrieval_embeddings = get_retrieval_embeddings(
        args.hf,
        args.hyperbolic,
        args.hybrid,
        args.model_path,
        args.base_model_type,
        device=args.device,
        batch_size=args.embed_batch_size
    )

    # Build corpus and retriever using custom embeddings
    print("Building document corpus...")
    corpus = build_document_corpus(raw_ds)
    
    print(f"Creating {'hyperbolic' if args.hyperbolic else 'Euclidean'} retriever...")
    retriever = create_retriever(
        embeddings=retrieval_embeddings,
        documents=corpus,
        top_k=args.top_k,
        is_hyperbolic=args.hyperbolic,
        distance_type=args.hyp_distance if args.hyperbolic else None
    )

    print(f"Initializing Qwen3 LLM: {args.qwen_model}")
    llm_evaluator, generation_pipeline = build_qwen3_ragas_llm(
        model_id=args.qwen_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Generate responses with RAG (using custom embeddings for retrieval)
    print("Running RAG pipeline (retrieve + generate)...")
    ragas_eval_ds = create_rag_dataset(
        hf_ds=raw_ds,
        retriever=retriever,
        generation_pipeline=generation_pipeline,
    )

    # Get RAGAS evaluation embeddings (always Qwen)
    print("Initializing RAGAS evaluation embeddings (Qwen3-Embedding-0.6B)...")
    ragas_embeddings = get_ragas_embeddings(
        device=args.device,
        batch_size=args.embed_batch_size
    )

    metrics = [Faithfulness(), ContextRelevance(), AnswerRelevancy()]

    print("Running RAGAS evaluation on generated responses...")
    run_evaluation(
        ragas_ds=ragas_eval_ds,
        embeddings_model=ragas_embeddings,  # Use Qwen embeddings for evaluation
        evaluator_llm=llm_evaluator,
        metrics=metrics,
        output_path=out_folder,
    )