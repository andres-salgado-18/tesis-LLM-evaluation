import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def compute_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduces dimensionality using PCA.

    Args:
        embeddings (np.ndarray): Input data of shape (N, D).
        n_components (int): Number of components to keep.

    Returns:
        np.ndarray: Reduced data of shape (N, n_components).
    """
    return PCA(n_components=n_components).fit_transform(embeddings)

def compute_pca_for_target(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduces dimensionality using PCA. If n_components is None, 
    it defaults to 0.95 variance retention.
    """

    pca_temp = PCA().fit(embeddings)
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Se seleccionaron {n_components} para 95% de varianza.")
    
    return PCA(n_components=n_components).fit_transform(embeddings)


def compute_tsne(embeddings: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """
    Reduces dimensionality using t-SNE.

    Args:
        embeddings (np.ndarray): Input data of shape (N, D).
        perplexity (float): Perplexity parameter for t-SNE.

    Returns:
        np.ndarray: Reduced data of shape (N, 2).
    """
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)

def compute_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduces dimensionality using UMAP.

    Args:
        embeddings (np.ndarray): Input data of shape (N, D).

    Returns:
        np.ndarray: Reduced data of shape (N, 2).
    """
    return umap.UMAP(random_state=42).fit_transform(embeddings)