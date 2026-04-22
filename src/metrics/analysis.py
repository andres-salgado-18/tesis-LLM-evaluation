import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from src.metrics.plots import plot_cosine_matrix, plot_model_bars
def get_similarity_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity matrix.
    Ensures normalization so that the dot product equals the cosine similarity.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    embeddings_norm = embeddings / (norms + 1e-9) #div 0
    return embeddings_norm @ embeddings_norm.T

def avg_similarity(indices: List[int], sim_matrix: np.ndarray) -> float:
    """Calculates average similarity within the same group (Intra-class)."""
    vals = [sim_matrix[i, j] for i in indices for j in indices if i < j]
    return float(np.mean(vals)) if vals else 0.0

def avg_similarity_between(indices1: List[int], indices2: List[int], sim_matrix: np.ndarray) -> float:
    """Calculates average similarity between different groups (Inter-class)."""
    vals = [sim_matrix[i, j] for i in indices1 for j in indices2]
    return float(np.mean(vals)) if vals else 0.0

def rank_dimensions_effect(
    df: pd.DataFrame, 
    sim_matrix: np.ndarray, 
    dimensions: List[str]
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Ranks dimensions based on how much they separate the latent space.
    Higher 'impact' means the dimension creates tighter, more distinct clusters.
    """
    results = {}

    for dim in dimensions:
        groups = df.groupby(dim).indices  
        keys = list(groups.keys())

        # 1. Intra-group similarity (consistency within the same label)
        intra_vals = [avg_similarity(groups[k], sim_matrix) for k in keys]
        
        # 2. Inter-group similarity (similarity across different labels)
        inter_vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter_vals.append(
                    avg_similarity_between(groups[keys[i]], groups[keys[j]], sim_matrix)
                )

        intra = float(np.mean(intra_vals)) if intra_vals else 0.0
        inter = float(np.mean(inter_vals)) if inter_vals else 0.0

        results[dim] = {
            "intra": intra,
            "inter": inter,
            "impact": intra - inter 
        }

    return sorted(results.items(), key=lambda x: x[1]["impact"], reverse=True)

def main():
    base_path = './outputs'  # Where your .parquet files are located
    results_dir = Path('./analysis')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = results_dir / "analysis_results.txt"
    
    # TARGET 
    target_col = "embedding_response" 
    
    dimensions = ["lang", "prompt_structure", "entity_familiarity", "ethical_pressure", "context_coherence"]
    
    files = [x for x in os.listdir(base_path) if x.endswith('embeddings.parquet')]

    for file_name in files:
        df = pd.read_parquet(os.path.join(base_path, file_name))
        model_name = file_name.replace("_embeddings.parquet", "")
        
        all_scenarios_results = {}

        for scenario, df_s in df.groupby("scenario"):
            df_s = df_s.sort_values(dimensions).reset_index(drop=True)
            embeddings = np.stack(df_s[target_col].values)
            cosine_matrix = get_similarity_cosine_matrix(embeddings)
            
            ranking = rank_dimensions_effect(df_s, cosine_matrix, dimensions)
            
            all_scenarios_results[scenario] = {dim: stats for dim, stats in ranking}
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"\nScenario: {scenario}\n")
                for dim, stats in ranking:
                    line = f"{dim:20}: impact={stats['impact']:.4f} (intra={stats['intra']:.3f}, inter={stats['inter']:.3f})\n"
                    print(line, end="")
                    f.write(line)
                f.write("-" * 30 + "\n")

            print("\n" + "="*40)


        plot_model_bars(all_scenarios_results, dimensions, model_name)
    
       

if __name__ == "__main__":
    main()