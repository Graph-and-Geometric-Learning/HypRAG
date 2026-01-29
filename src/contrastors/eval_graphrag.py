import os
import json
import argparse
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import torch
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

# For community detection
try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("Warning: leidenalg not available, falling back to Louvain community detection")


# -------------------------
# Args / Env
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GraphRAG Evaluation with RAGAS")
    
    # Model settings
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace sentence transformer model for embeddings")
    parser.add_argument("--llm_model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model for entity extraction and response generation")
    
    # Dataset settings
    parser.add_argument("--dataset", required=True, help="RAGBench config name (e.g., delucionqa)")
    parser.add_argument("--split", default="test", help="HF split (default: test)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Max samples to evaluate (for testing)")
    
    # GraphRAG settings
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for documents")
    parser.add_argument("--chunk_overlap", type=int, default=64, help="Overlap between chunks")
    parser.add_argument("--community_level", type=int, default=1,
                        help="Community hierarchy level for global search (0=finest)")
    parser.add_argument("--top_k_entities", type=int, default=10,
                        help="Number of top entities to retrieve for local search")
    parser.add_argument("--top_k_communities", type=int, default=5,
                        help="Number of top communities for global search")
    parser.add_argument("--search_type", default="local", choices=["local", "global", "hybrid"],
                        help="Type of GraphRAG search")
    
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    # Hardware settings
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    
    # Caching
    parser.add_argument("--cache_dir", default="./graphrag_cache",
                        help="Directory to cache graph index")
    parser.add_argument("--force_reindex", action="store_true",
                        help="Force re-indexing even if cache exists")

    return parser.parse_args()


def validate_env():
    if "HF_TOKEN" not in os.environ:
        raise EnvironmentError("HF_TOKEN must be set.")


# -------------------------
# Embeddings Wrapper
# -------------------------
class SentenceTransformerEmbeddings(BaseLangchainEmbeddings):
    """LangChain-compatible wrapper for SentenceTransformers"""
    
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64, normalize: bool = True):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()


# -------------------------
# Text Chunking
# -------------------------
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "start_idx": start,
            "end_idx": min(end, len(words))
        })
        
        chunk_id += 1
        start += chunk_size - overlap
        
        if end >= len(words):
            break
    
    return chunks


# -------------------------
# Entity/Relationship Extraction
# -------------------------
ENTITY_EXTRACTION_PROMPT = """Extract all entities and relationships from the following text.

TEXT:
{text}

Output your response in the following JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|OTHER", "description": "brief description"}}
    ],
    "relationships": [
        {{"source": "entity1 name", "target": "entity2 name", "relationship": "relationship type", "description": "brief description"}}
    ]
}}

Only output valid JSON, no other text."""


class EntityExtractor:
    """Extract entities and relationships from text using an LLM"""
    
    def __init__(self, llm_pipeline, max_retries: int = 3):
        self.llm_pipeline = llm_pipeline
        self.max_retries = max_retries
    
    def extract(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_pipeline(messages, max_new_tokens=1024)[0]['generated_text']
                
                # Try to parse JSON
                # Find JSON in response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)
                    
                    entities = data.get("entities", [])
                    relationships = data.get("relationships", [])
                    
                    return entities, relationships
                    
            except (json.JSONDecodeError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: Failed to extract entities after {self.max_retries} attempts")
                    return [], []
        
        return [], []


# -------------------------
# Knowledge Graph
# -------------------------
class KnowledgeGraph:
    """Knowledge graph built from extracted entities and relationships"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_descriptions: Dict[str, str] = {}
        self.entity_types: Dict[str, str] = {}
        self.relationship_descriptions: Dict[Tuple[str, str], str] = {}
        self.entity_to_chunks: Dict[str, List[int]] = {}  # Maps entity to chunk IDs
        self.chunks: List[Dict] = []
        self.communities: Dict[int, List[str]] = {}  # community_id -> [entity_names]
        self.community_summaries: Dict[int, str] = {}
        
    def add_entity(self, name: str, entity_type: str, description: str, chunk_id: int):
        """Add an entity to the graph"""
        name_lower = name.lower().strip()
        
        if name_lower not in self.graph:
            self.graph.add_node(name_lower)
            self.entity_descriptions[name_lower] = description
            self.entity_types[name_lower] = entity_type
            self.entity_to_chunks[name_lower] = []
        
        if chunk_id not in self.entity_to_chunks[name_lower]:
            self.entity_to_chunks[name_lower].append(chunk_id)
    
    def add_relationship(self, source: str, target: str, rel_type: str, description: str):
        """Add a relationship to the graph"""
        source_lower = source.lower().strip()
        target_lower = target.lower().strip()
        
        if source_lower in self.graph and target_lower in self.graph:
            self.graph.add_edge(source_lower, target_lower, relationship=rel_type)
            self.relationship_descriptions[(source_lower, target_lower)] = description
    
    def add_chunk(self, chunk: Dict):
        """Add a text chunk"""
        self.chunks.append(chunk)
    
    def detect_communities(self) -> Dict[int, List[str]]:
        """Detect communities in the graph using Leiden or Louvain algorithm"""
        if len(self.graph.nodes()) == 0:
            return {}
        
        if LEIDEN_AVAILABLE:
            # Convert to igraph for Leiden
            ig_graph = ig.Graph.from_networkx(self.graph)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            
            communities = {}
            node_list = list(self.graph.nodes())
            for idx, community in enumerate(partition):
                communities[idx] = [node_list[i] for i in community]
        else:
            # Fallback to Louvain
            from networkx.algorithms.community import louvain_communities
            partitions = louvain_communities(self.graph, seed=42)
            communities = {idx: list(comm) for idx, comm in enumerate(partitions)}
        
        self.communities = communities
        return communities
    
    def get_entity_context(self, entity_name: str) -> str:
        """Get context for an entity including its description and relationships"""
        entity = entity_name.lower().strip()
        
        if entity not in self.graph:
            return ""
        
        context_parts = []
        
        # Entity description
        if entity in self.entity_descriptions:
            context_parts.append(f"Entity: {entity}")
            context_parts.append(f"Type: {self.entity_types.get(entity, 'Unknown')}")
            context_parts.append(f"Description: {self.entity_descriptions[entity]}")
        
        # Relationships
        neighbors = list(self.graph.neighbors(entity))
        if neighbors:
            context_parts.append("Relationships:")
            for neighbor in neighbors[:10]:  # Limit relationships
                edge_data = self.graph.get_edge_data(entity, neighbor)
                rel_type = edge_data.get('relationship', 'related to') if edge_data else 'related to'
                context_parts.append(f"  - {entity} {rel_type} {neighbor}")
        
        return "\n".join(context_parts)
    
    def get_chunks_for_entity(self, entity_name: str) -> List[Dict]:
        """Get all chunks that mention an entity"""
        entity = entity_name.lower().strip()
        chunk_ids = self.entity_to_chunks.get(entity, [])
        return [self.chunks[cid] for cid in chunk_ids if cid < len(self.chunks)]
    
    def save(self, path: str):
        """Save the knowledge graph to disk"""
        data = {
            'graph': nx.node_link_data(self.graph),
            'entity_descriptions': self.entity_descriptions,
            'entity_types': self.entity_types,
            'relationship_descriptions': {f"{k[0]}|||{k[1]}": v for k, v in self.relationship_descriptions.items()},
            'entity_to_chunks': self.entity_to_chunks,
            'chunks': self.chunks,
            'communities': self.communities,
            'community_summaries': self.community_summaries,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load a knowledge graph from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        kg.graph = nx.node_link_graph(data['graph'])
        kg.entity_descriptions = data['entity_descriptions']
        kg.entity_types = data['entity_types']
        kg.relationship_descriptions = {tuple(k.split("|||")): v for k, v in data['relationship_descriptions'].items()}
        kg.entity_to_chunks = data['entity_to_chunks']
        kg.chunks = data['chunks']
        kg.communities = data.get('communities', {})
        kg.community_summaries = data.get('community_summaries', {})
        
        return kg


# -------------------------
# Community Summarization
# -------------------------
COMMUNITY_SUMMARY_PROMPT = """Summarize the following group of related entities and their relationships.
This summary should capture the main themes, key entities, and how they relate to each other.

ENTITIES AND RELATIONSHIPS:
{content}

Provide a concise summary (2-3 paragraphs) that captures the key information about this group."""


def generate_community_summaries(kg: KnowledgeGraph, llm_pipeline) -> Dict[int, str]:
    """Generate summaries for each community"""
    summaries = {}
    
    for comm_id, entities in tqdm(kg.communities.items(), desc="Generating community summaries"):
        if not entities:
            continue
        
        # Build content for this community
        content_parts = []
        
        for entity in entities[:20]:  # Limit entities per community
            entity_ctx = kg.get_entity_context(entity)
            if entity_ctx:
                content_parts.append(entity_ctx)
        
        if not content_parts:
            continue
        
        content = "\n\n".join(content_parts)
        prompt = COMMUNITY_SUMMARY_PROMPT.format(content=content)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = llm_pipeline(messages, max_new_tokens=512)[0]['generated_text']
            summaries[comm_id] = response.strip()
        except Exception as e:
            print(f"Warning: Failed to generate summary for community {comm_id}: {e}")
            summaries[comm_id] = f"Community containing: {', '.join(entities[:10])}"
    
    kg.community_summaries = summaries
    return summaries


# -------------------------
# GraphRAG Indexer
# -------------------------
class GraphRAGIndexer:
    """Build a GraphRAG index from documents"""
    
    def __init__(
        self,
        embeddings: SentenceTransformerEmbeddings,
        llm_pipeline,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.embeddings = embeddings
        self.llm_pipeline = llm_pipeline
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.entity_extractor = EntityExtractor(llm_pipeline)
        self.kg = KnowledgeGraph()
        
        # Embeddings storage
        self.chunk_embeddings: np.ndarray = None
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.community_embeddings: Dict[int, np.ndarray] = {}
    
    def index(self, documents: List[str], show_progress: bool = True) -> KnowledgeGraph:
        """Index a list of documents"""
        
        # 1. Chunk all documents
        all_chunks = []
        for doc_id, doc in enumerate(tqdm(documents, desc="Chunking documents", disable=not show_progress)):
            chunks = chunk_text(doc, self.chunk_size, self.chunk_overlap)
            for chunk in chunks:
                chunk['doc_id'] = doc_id
                chunk['global_id'] = len(all_chunks)
                all_chunks.append(chunk)
                self.kg.add_chunk(chunk)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # 2. Extract entities and relationships from each chunk
        print("Extracting entities and relationships...")
        for chunk in tqdm(all_chunks, desc="Entity extraction", disable=not show_progress):
            entities, relationships = self.entity_extractor.extract(chunk['text'])
            
            for entity in entities:
                self.kg.add_entity(
                    name=entity.get('name', ''),
                    entity_type=entity.get('type', 'OTHER'),
                    description=entity.get('description', ''),
                    chunk_id=chunk['global_id']
                )
            
            for rel in relationships:
                self.kg.add_relationship(
                    source=rel.get('source', ''),
                    target=rel.get('target', ''),
                    rel_type=rel.get('relationship', 'related'),
                    description=rel.get('description', '')
                )
        
        print(f"Extracted {len(self.kg.graph.nodes())} entities and {len(self.kg.graph.edges())} relationships")
        
        # 3. Detect communities
        print("Detecting communities...")
        communities = self.kg.detect_communities()
        print(f"Detected {len(communities)} communities")
        
        # 4. Generate community summaries
        print("Generating community summaries...")
        generate_community_summaries(self.kg, self.llm_pipeline)
        
        # 5. Compute embeddings
        print("Computing embeddings...")
        self._compute_embeddings()
        
        return self.kg
    
    def _compute_embeddings(self):
        """Compute embeddings for chunks, entities, and communities"""
        
        # Chunk embeddings
        chunk_texts = [c['text'] for c in self.kg.chunks]
        if chunk_texts:
            self.chunk_embeddings = np.array(self.embeddings.embed_documents(chunk_texts))
        
        # Entity embeddings (based on description + context)
        entity_texts = []
        entity_names = list(self.kg.graph.nodes())
        for entity in entity_names:
            ctx = self.kg.get_entity_context(entity)
            entity_texts.append(ctx if ctx else entity)
        
        if entity_texts:
            entity_embs = np.array(self.embeddings.embed_documents(entity_texts))
            self.entity_embeddings = {name: emb for name, emb in zip(entity_names, entity_embs)}
        
        # Community embeddings (based on summaries)
        for comm_id, summary in self.kg.community_summaries.items():
            if summary:
                emb = np.array(self.embeddings.embed_query(summary))
                self.community_embeddings[comm_id] = emb
    
    def save(self, path: str):
        """Save the index to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save knowledge graph
        self.kg.save(os.path.join(path, "knowledge_graph.pkl"))
        
        # Save embeddings
        np.save(os.path.join(path, "chunk_embeddings.npy"), self.chunk_embeddings)
        
        with open(os.path.join(path, "entity_embeddings.pkl"), 'wb') as f:
            pickle.dump(self.entity_embeddings, f)
        
        with open(os.path.join(path, "community_embeddings.pkl"), 'wb') as f:
            pickle.dump(self.community_embeddings, f)
    
    @classmethod
    def load(cls, path: str, embeddings: SentenceTransformerEmbeddings, llm_pipeline) -> 'GraphRAGIndexer':
        """Load an index from disk"""
        indexer = cls(embeddings, llm_pipeline)
        
        indexer.kg = KnowledgeGraph.load(os.path.join(path, "knowledge_graph.pkl"))
        indexer.chunk_embeddings = np.load(os.path.join(path, "chunk_embeddings.npy"))
        
        with open(os.path.join(path, "entity_embeddings.pkl"), 'rb') as f:
            indexer.entity_embeddings = pickle.load(f)
        
        with open(os.path.join(path, "community_embeddings.pkl"), 'rb') as f:
            indexer.community_embeddings = pickle.load(f)
        
        return indexer


# -------------------------
# GraphRAG Retriever
# -------------------------
class GraphRAGRetriever:
    """Retrieve relevant context using GraphRAG"""
    
    def __init__(
        self,
        indexer: GraphRAGIndexer,
        search_type: str = "local",
        top_k_entities: int = 10,
        top_k_communities: int = 5,
    ):
        self.indexer = indexer
        self.search_type = search_type
        self.top_k_entities = top_k_entities
        self.top_k_communities = top_k_communities
    
    def _local_search(self, query: str) -> List[str]:
        """Local search: find relevant entities and their context"""
        query_emb = np.array(self.indexer.embeddings.embed_query(query))
        
        # Find top-k similar entities
        entity_names = list(self.indexer.entity_embeddings.keys())
        if not entity_names:
            return self._fallback_chunk_search(query)
        
        entity_embs = np.array([self.indexer.entity_embeddings[e] for e in entity_names])
        
        # Cosine similarity
        similarities = np.dot(entity_embs, query_emb)
        top_indices = np.argsort(similarities)[-self.top_k_entities:][::-1]
        
        contexts = []
        seen_chunks = set()
        
        for idx in top_indices:
            entity_name = entity_names[idx]
            
            # Get entity context
            entity_ctx = self.indexer.kg.get_entity_context(entity_name)
            if entity_ctx:
                contexts.append(entity_ctx)
            
            # Get related chunks
            for chunk in self.indexer.kg.get_chunks_for_entity(entity_name):
                if chunk['global_id'] not in seen_chunks:
                    contexts.append(chunk['text'])
                    seen_chunks.add(chunk['global_id'])
        
        return contexts[:self.top_k_entities * 2]  # Limit total contexts
    
    def _global_search(self, query: str) -> List[str]:
        """Global search: find relevant communities and their summaries"""
        query_emb = np.array(self.indexer.embeddings.embed_query(query))
        
        if not self.indexer.community_embeddings:
            return self._fallback_chunk_search(query)
        
        comm_ids = list(self.indexer.community_embeddings.keys())
        comm_embs = np.array([self.indexer.community_embeddings[c] for c in comm_ids])
        
        # Cosine similarity
        similarities = np.dot(comm_embs, query_emb)
        top_indices = np.argsort(similarities)[-self.top_k_communities:][::-1]
        
        contexts = []
        for idx in top_indices:
            comm_id = comm_ids[idx]
            summary = self.indexer.kg.community_summaries.get(comm_id, "")
            if summary:
                contexts.append(f"Community Summary:\n{summary}")
            
            # Also add some entity details from this community
            entities = self.indexer.kg.communities.get(comm_id, [])
            for entity in entities[:3]:
                entity_ctx = self.indexer.kg.get_entity_context(entity)
                if entity_ctx:
                    contexts.append(entity_ctx)
        
        return contexts
    
    def _hybrid_search(self, query: str) -> List[str]:
        """Hybrid search: combine local and global search"""
        local_contexts = self._local_search(query)
        global_contexts = self._global_search(query)
        
        # Interleave results
        combined = []
        for i in range(max(len(local_contexts), len(global_contexts))):
            if i < len(global_contexts):
                combined.append(global_contexts[i])
            if i < len(local_contexts):
                combined.append(local_contexts[i])
        
        return combined[:self.top_k_entities + self.top_k_communities]
    
    def _fallback_chunk_search(self, query: str) -> List[str]:
        """Fallback to simple chunk-based semantic search"""
        if self.indexer.chunk_embeddings is None or len(self.indexer.chunk_embeddings) == 0:
            return []
        
        query_emb = np.array(self.indexer.embeddings.embed_query(query))
        similarities = np.dot(self.indexer.chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[-self.top_k_entities:][::-1]
        
        return [self.indexer.kg.chunks[idx]['text'] for idx in top_indices]
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant contexts for a query"""
        if self.search_type == "local":
            return self._local_search(query)
        elif self.search_type == "global":
            return self._global_search(query)
        elif self.search_type == "hybrid":
            return self._hybrid_search(query)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")


# -------------------------
# Response Generation
# -------------------------
RESPONSE_GENERATION_PROMPT = """Based on the following context, answer the question. 
Use the information provided to give a comprehensive and accurate answer.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def generate_response(query: str, contexts: List[str], llm_pipeline) -> str:
    """Generate a response given query and retrieved contexts"""
    context_str = "\n\n---\n\n".join(contexts)
    
    prompt = RESPONSE_GENERATION_PROMPT.format(context=context_str, question=query)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = llm_pipeline(messages, max_new_tokens=512)[0]['generated_text']
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I could not generate a response based on the provided context."


# -------------------------
# Dataset Loading
# -------------------------
def load_ragbench_split(name: str, split: str = "test") -> Dataset:
    return load_dataset("galileo-ai/ragbench", name, split=split)


def build_document_corpus(hf_ds: Dataset) -> List[str]:
    """Extract all unique documents from the dataset"""
    all_docs = set()
    
    doc_col = None
    for col in ["documents", "context", "retrieved_contexts"]:
        if col in hf_ds.column_names:
            doc_col = col
            break
    
    if not doc_col:
        raise ValueError(f"No documents column found in {hf_ds.column_names}")
    
    for row in hf_ds:
        docs = row[doc_col]
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


def get_corpus_hash(corpus: List[str]) -> str:
    """Generate a hash for the corpus to check if it changed"""
    content = "".join(sorted(corpus))
    return hashlib.md5(content.encode()).hexdigest()[:12]


# -------------------------
# LLM Setup
# -------------------------
def build_llm_pipeline(model_id: str, max_new_tokens: int = 512, temperature: float = 0.5):
    """Build HuggingFace pipeline for text generation"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )
    
    return pipe


def build_ragas_llm(llm_pipeline):
    """Wrap pipeline for RAGAS"""
    hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return LangchainLLMWrapper(hf_llm)


# -------------------------
# RAG Dataset Creation
# -------------------------
def create_graphrag_dataset(
    hf_ds: Dataset,
    retriever: GraphRAGRetriever,
    llm_pipeline,
    max_samples: Optional[int] = None,
) -> EvaluationDataset:
    """Create RAGAS evaluation dataset using GraphRAG retrieval"""
    
    rag_data = []
    query_col = "question" if "question" in hf_ds.column_names else "query"
    
    n_samples = len(hf_ds) if max_samples is None else min(max_samples, len(hf_ds))
    
    print(f"Generating responses for {n_samples} queries...")
    for i in tqdm(range(n_samples), desc="Processing queries"):
        row = hf_ds[i]
        query = row[query_col]
        
        # Retrieve with GraphRAG
        retrieved_contexts = retriever.retrieve(query)
        
        # Generate response
        response = generate_response(query, retrieved_contexts, llm_pipeline)
        
        rag_data.append({
            "user_input": query,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        })
    
    rag_hf_ds = Dataset.from_list(rag_data)
    return EvaluationDataset.from_hf_dataset(rag_hf_ds)


# -------------------------
# Evaluation
# -------------------------
def run_evaluation(
    ragas_ds: EvaluationDataset,
    embeddings_model,
    evaluator_llm,
    metrics,
    output_path: str,
):
    """Run RAGAS evaluation"""
    run_config = RunConfig(
        max_workers=1,
        timeout=1200,
        max_retries=5,
        max_wait=180,
        exception_types=(Exception,),
    )
    
    results = evaluate(
        dataset=ragas_ds,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings_model,
        run_config=run_config,
        raise_exceptions=False,
    )
    
    scores_df = results.to_pandas()
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed scores
    scores_path = os.path.join(output_path, "ragas_scores.jsonl")
    with open(scores_path, "w") as f:
        for _, row in scores_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    
    # Save summary
    summary = scores_df.mean(numeric_only=True).to_dict()
    summary_path = os.path.join(output_path, "ragas_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSaved detailed results to: {scores_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
    
    return scores_df


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    validate_env()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(args.output, f"graphrag_{args.dataset}_{args.search_type}_{run_id}")
    
    # Initialize embeddings
    print(f"Initializing embeddings: {args.embedding_model}")
    embeddings = SentenceTransformerEmbeddings(
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.embed_batch_size,
    )
    
    # Initialize LLM
    print(f"Initializing LLM: {args.llm_model}")
    llm_pipeline = build_llm_pipeline(
        args.llm_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Load dataset
    print(f"Loading RAGBench: {args.dataset} / {args.split}")
    raw_ds = load_ragbench_split(args.dataset, args.split)
    
    # Build corpus
    corpus = build_document_corpus(raw_ds)
    corpus_hash = get_corpus_hash(corpus)
    
    # Check cache
    cache_path = os.path.join(args.cache_dir, f"{args.dataset}_{corpus_hash}")
    
    if os.path.exists(cache_path) and not args.force_reindex:
        print(f"Loading cached index from {cache_path}")
        indexer = GraphRAGIndexer.load(cache_path, embeddings, llm_pipeline)
    else:
        print("Building GraphRAG index (this may take a while)...")
        indexer = GraphRAGIndexer(
            embeddings=embeddings,
            llm_pipeline=llm_pipeline,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        indexer.index(corpus)
        
        # Save cache
        print(f"Saving index to {cache_path}")
        indexer.save(cache_path)
    
    # Create retriever
    retriever = GraphRAGRetriever(
        indexer=indexer,
        search_type=args.search_type,
        top_k_entities=args.top_k_entities,
        top_k_communities=args.top_k_communities,
    )
    
    # Generate responses with GraphRAG
    print("Creating GraphRAG evaluation dataset...")
    ragas_ds = create_graphrag_dataset(
        hf_ds=raw_ds,
        retriever=retriever,
        llm_pipeline=llm_pipeline,
        max_samples=args.max_samples,
    )
    
    # RAGAS evaluation
    print("Initializing RAGAS evaluation...")
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    ragas_llm = build_ragas_llm(llm_pipeline)
    
    metrics = [Faithfulness(), ContextRelevance(), AnswerRelevancy()]
    
    print("Running RAGAS evaluation...")
    run_evaluation(
        ragas_ds=ragas_ds,
        embeddings_model=ragas_embeddings,
        evaluator_llm=ragas_llm,
        metrics=metrics,
        output_path=output_folder,
    )
    
    # Save config
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nDone! Results saved to {output_folder}")


if __name__ == "__main__":
    main()