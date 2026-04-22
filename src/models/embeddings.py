import numpy as np
from src.models.base import BaseEmbedder
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class LocalEmbedder(BaseEmbedder):
    """
    Implementation of BaseEmbedder using local Sentence-Transformer models.
    Runs locally on CPU or GPU using the sentence-transformers library.
    """
    def __init__(self, name: str):
        """
        Initializes the local transformer model.

        Args:
            name (str): The model identifier from HuggingFace (e.g., 'all-MiniLM-L6-v2').
        """
        self._name = name
        self.model = SentenceTransformer(name)
    
    @property
    def model_name(self) -> str:
        """Returns the name of the local model."""
        return self._name

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generates normalized embeddings for a list of texts using the local model.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            np.ndarray: A 2D array of normalized embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

class OpenAIEmbedder(BaseEmbedder):
    """
    Implementation of BaseEmbedder using the OpenAI Embeddings API.
    Requires valid API credentials.
    """
    def __init__(self, client: OpenAI, model="text-embedding-3-small"):
        """
        Initializes the OpenAI embedder client.

        Args:
            client (OpenAI): An instance of the OpenAI python client.
            model (str): The OpenAI model ID. Defaults to 'text-embedding-3-small'.
        """
        self.client = client
        self._name = model

    @property
    def model_name(self) -> str:
        """Returns the name of the OpenAI model."""
        return self._name
    
    def embed(self, texts):
        """
        Generates embeddings for a list of texts by calling the OpenAI API.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            np.ndarray: A 2D array of embeddings retrieved from the API.
        """
        response = self.client.embeddings.create(
            model=self._name,
            input=texts
        )
        return np.array([item.embedding for item in response.data])