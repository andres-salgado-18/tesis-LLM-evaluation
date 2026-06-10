import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from itertools import combinations
def cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Computes the cosine distance (1 - similarity) between two embedding vectors.

    Args:
        e1 (np.ndarray): First embedding vector (D,).
        e2 (np.ndarray): Second embedding vector (D,).
    
    Returns:
        float: Cosine distance between e1 and e2.
    """
    similarity = cosine_similarity(
        [e1],
        [e2]
    )[0][0]

    return 1 - similarity


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise cosine similarity matrix for a set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D array of shape (N, D) where N is 
                                 the number of samples and D is the 
                                 embedding dimension.

    Returns:
        np.ndarray: A similarity matrix of shape (N, N) where each 
                    element [i, j] represents the cosine similarity 
                    between embedding i and embedding j.
    """
    return cosine_similarity(
        embeddings
    )

def calculate_dispersion(group_embeddings: np.ndarray) -> float:
    """
    Computes average dispersion between uniques pair of embeddings.
    Args:
        group_embeddings: Embedding array (n_runs, embedding_dim).

    Returns:
        float: Average cosine distance between pairs.
    """
    distances = [
        cosine_distance(emb1, emb2) 
        for emb1, emb2 in combinations(group_embeddings, 2)
    ]
    return float(np.mean(distances))

