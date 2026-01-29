import networkx as nx
import numpy as np
import argparse
import logging
import os

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_texts(num_samples=50000):
    """Load first num_samples texts from MSMARCO dataset."""
    logger.info(f"Loading {num_samples} texts from mteb/msmarco...")
    dataset = load_dataset("mteb/msmarco", "corpus", split="corpus")
    texts = dataset["text"][:num_samples]
    return texts


def generate_embeddings(texts, model_name, batch_size=128):
    """Generate embeddings using SentenceTransformer."""
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def build_knn_graph(embeddings, k=10):
    """Build k-NN graph from embeddings."""
    logger.info(f"Building {k}-NN graph...")
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    
    # Get k nearest neighbors (excluding self)
    distances, indices = nn.kneighbors(embeddings)
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings)))
    
    for i in range(len(embeddings)):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):  # skip self (index 0)
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=1 - dist)  # convert distance to similarity
    
    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def compute_ricci_curvature(G, alpha=0.5):
    """Compute Ollivier-Ricci curvature for the graph."""
    logger.info("Computing Ollivier-Ricci curvature...")
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO")
    orc.compute_ricci_curvature()
    return orc.G.copy()


def save_results(G_orc, model_name, output_dir="data/graphs"):
    """Save Ricci curvatures and plot histogram."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract curvatures
    ricci_curvatures = np.array(list(nx.get_edge_attributes(G_orc, "ricciCurvature").values()))
    
    # Create safe filename from model name
    safe_name = model_name.replace("/", "_").replace("-", "_")
    
    # Save curvatures
    npy_path = os.path.join(output_dir, f"{safe_name}_msmarco_ricci.npy")
    np.save(npy_path, ricci_curvatures)
    logger.info(f"Saved curvatures to {npy_path}")
    
    # Save graph
    graphml_path = os.path.join(output_dir, f"{safe_name}_msmarco_knn_graph.graphml")
    nx.write_graphml(G_orc, graphml_path)
    logger.info(f"Saved graph to {graphml_path}")
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Mean curvature: {ricci_curvatures.mean():.6f}")
    print(f"Std curvature: {ricci_curvatures.std():.6f}")
    print(f"Min curvature: {ricci_curvatures.min():.6f}")
    print(f"Max curvature: {ricci_curvatures.max():.6f}")
    print(f"{'='*50}\n")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ricci_curvatures, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Ricci Curvature')
    plt.ylabel('Frequency')
    plt.title(f"Histogram of Ricci Curvatures\n({model_name} on MSMARCO, k=10)")
    plt.axvline(x=ricci_curvatures.mean(), color='r', linestyle='--', label=f'Mean: {ricci_curvatures.mean():.4f}')
    plt.legend()
    
    fig_path = os.path.join(output_dir, f"{safe_name}_msmarco_ricci_hist.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved histogram to {fig_path}")
    
    return ricci_curvatures


def main():
    parser = argparse.ArgumentParser(description="Compute Ricci curvature on k-NN graph from MSMARCO embeddings")
    parser.add_argument("model", type=str, help="HuggingFace model name for SentenceTransformer")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of texts to use")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for Ollivier-Ricci")
    parser.add_argument("--output_dir", type=str, default="logs/curvature/", help="Output directory")
    
    args = parser.parse_args()
    
    # Load texts
    texts = load_texts(args.num_samples)
    
    # Generate embeddings
    embeddings = generate_embeddings(texts, args.model, args.batch_size)
    
    # Build k-NN graph
    G = build_knn_graph(embeddings, args.k)
    G = G.subgraph(list(range(25000, 75000)))
    # Compute Ricci curvature
    G_orc = compute_ricci_curvature(G, args.alpha)
    
    # Save results
    ricci_curvatures = save_results(G_orc, args.model, args.output_dir)
    
    return ricci_curvatures


if __name__ == "__main__":
    main()
