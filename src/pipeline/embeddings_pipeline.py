import numpy as np
import pandas as pd
from src.models.embeddings import OpenAIEmbedder
from pathlib import Path


class EmbeddingsPipeline:
    """
    Orchestrates the process of generating and storing embeddings for multiple DataFrame columns.
    
    This pipeline handles data cleaning (NaN handling), batching logic to optimize 
    API or hardware usage, and final storage of vectors in a parquet format.
    """
    def __init__(self, embedder):
        """
        Initializes the pipeline with a specific embedding implementation.

        Args:
            embedder (BaseEmbedder): An instance of a class implementing the BaseEmbedder interface.
        """
        self.embedder = embedder

    def run(self, df: pd.DataFrame, config, base_path="outputs", text_cols=("prompt", "response")):
        """
        Executes the embedding process for the specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing text columns.
            config: Configuration object containing batch_size and .id() method.
            base_path (str): Directory where the final output will be saved.
            text_cols (tuple): Names of the columns to be vectorized.

        Returns:
            tuple: (pd.DataFrame, str) The DataFrame with added embedding columns and the output file path.
        
        Raises:
            Exception: Re-raises any exception encountered during the embedding process.
        """
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        df_out = df.copy()

        for col in text_cols:
            df_out[col] = df_out[col].fillna("").astype(str)
            texts = df_out[col].tolist()
            all_embeddings = []

            for i in range(0, len(texts), config.batch_size):
                batch = texts[i : i + config.batch_size]
                try:
                    emb = self.embedder.embed(batch)
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"Error en batch {i} de la columna {col}: {e}")
                    raise e
            
            embeddings = np.vstack(all_embeddings)
            df_out[f"embedding_{col}"] = list(embeddings)

        out_path = f"{base_path}/{config.id()}_embeddings.parquet"
        df_out.to_parquet(out_path, index=False)
        return df_out, out_path