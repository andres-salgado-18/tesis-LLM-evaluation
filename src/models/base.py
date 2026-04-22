from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Calculates embeddings for a batch of strings.
        
        Args:
            texts: A list of strings to be vectorized.
            
        Returns:
            np.ndarray: A 2D matrix with shape (n_samples, n_dimensions).
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the unique identifier of the model being used."""
        pass

class LLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a text completion based on the provided prompt.
        
        Args:
            prompt: The input text for the model.
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            str: The generated text response.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name of the generative model."""
        pass