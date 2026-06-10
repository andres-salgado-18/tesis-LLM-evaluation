import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_clusters(clusters: np.ndarray, labels: np.ndarray, dimension_name: str) -> None:
    """
    Prints clustering performance metrics (ARI, NMI).

    Args:
        clusters (np.ndarray): Predicted cluster labels.
        labels (np.ndarray): Ground truth labels.
        dimension_name (str): Identifier for the dimension being evaluated.
    """
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    
    print(f"\n--- Dimension: {dimension_name} ---")
    print(f"ARI: {ari:.4f} | NMI: {nmi:.4f}")
    return ari, nmi