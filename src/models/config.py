from dataclasses import dataclass

@dataclass
class OpenAIGenerationConfig:
    """
    Configuration for LLM generation tasks.
    
    Attributes:
        model (str): Name of the OpenAI model (e.g., 'gpt-4o-mini').
        temperature (float): Sampling temperature (0 to 2). Defaults to 0.1.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 500.
    """
    model: str
    temperature: float = 0.1
    max_tokens: int = 500

    def id(self):
        return f"{self.model}_t{self.temperature}_mt{self.max_tokens}"


@dataclass
class OpenAIEmbeddingConfig:
    """
    Configuration for OpenAI Embedding API tasks.
    
    Attributes:
        model (str): Name of the embedding model (e.g., 'text-embedding-3-small').
        batch_size (int): Number of texts sent per API request. Defaults to 64.
    """
    model: str
    batch_size: int = 64

    def id(self):
        return f"{self.model}_bs{self.batch_size}"

"""
Configuration for the local S-BERT models.
Attributes:
        model (str): Name of the embedding model identifier.
        batch_size (int): Number of texts processed per hardware iteration. Defaults to 64.
"""   
@dataclass
class all_MiniLM_L6_v2Config:
    model: str
    batch_size: int = 64

    def id(self):
        return f"{self.model}_bs{self.batch_size}" 
    
@dataclass
class all_mpnet_base_v2Config:
    model: str
    batch_size: int = 64

    def id(self):
        return f"{self.model}_bs{self.batch_size}" 
    
@dataclass
class paraphrase_multilingual_MiniLM_L12_v2Config:
    model: str
    batch_size: int = 64

    def id(self):
        return f"{self.model}_bs{self.batch_size}" 