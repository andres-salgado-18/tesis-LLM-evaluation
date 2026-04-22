import pandas as pd
import time
import json

class OpenAIBatchGenerator:
    """
    Handles the low-level mechanics of the OpenAI Batch API.
    
    This class is responsible for transforming local DataFrames into OpenAI-compatible 
    JSONL files, managing file uploads, and monitoring batch job status.
    """ 
    def __init__(self, client):
        """
        Initializes the generator with an OpenAI client instance.

        Args:
            client (OpenAI): The OpenAI Python client.
        """
        self.client = client

    def create_batch_file(
        self,
        df: pd.DataFrame,
        out_path: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        """
        Generates a JSONL file for the Batch API from a DataFrame.

        Args:
            df (pd.DataFrame): Input data containing 'index' and 'prompt' columns.
            out_path (str): Target path for the generated .jsonl file.
            model (str): Model identifier for the API request.
            temperature (float): Controls randomness.
            max_tokens (int): Maximum length of the generated response.

        Returns:
            str: The file path of the created JSONL.
        """

        with open(out_path, "w", encoding="utf-8") as f:
            for row in df.to_dict("records"):

                record = {
                    "custom_id":f"idx_{row['index']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": row["prompt"]}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return out_path

    def upload_file(self, path: str):
        """
        Uploads a local file to the OpenAI storage for batch processing.

        Args:
            path (str): Path to the .jsonl file.

        Returns:
            str: The unique file ID provided by OpenAI.
        """
        with open(path, "rb") as f:
            file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        return file.id

    def create_batch(self, file_id: str):
        """
        Triggers the creation of a batch job using a previously uploaded file.

        Args:
            file_id (str): The OpenAI ID of the input file.

        Returns:
            str: The unique batch job ID.
        """
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch.id

    def wait_batch(self, batch_id: str):
        """
        Blocks execution and polls the batch status until it reaches a terminal state.

        Uses an exponential backoff strategy for polling (5s to 30s).

        Args:
            batch_id (str): The ID of the batch to monitor.

        Returns:
            openai.types.Batch: The final batch object on success.

        Raises:
            Exception: If the batch fails, cancels, or expires.
        """
        delay = 5

        while True:
            batch = self.client.batches.retrieve(batch_id)

            print("[STATUS]", batch.status)

            if batch.status == "completed":
                return batch

            if batch.status in ["failed", "expired"]:
                raise Exception(f"Batch failed: {batch.status}")

            time.sleep(delay)
            delay = min(delay * 1.5, 30)

    def download_results(self, batch, out_path: str = "responses.jsonl") -> str:
        """
        Retrieves and saves the results from a completed batch job.

        Args:
            batch: The completed OpenAI batch object.
            out_path (str): Path where the responses will be saved.

        Returns:
            str: The path to the downloaded responses file.
        """"""
        Retrieves and saves the results from a completed batch job.

        Args:
            batch: The completed OpenAI batch object.
            out_path (str): Path where the responses will be saved.

        Returns:
            str: The path to the downloaded responses file.
        """
        file_id = batch.output_file_id
        if file_id is None:
            raise Exception("No output file")

        
        content = self.client.files.content(file_id).content 

        with open(out_path, "wb") as f:
            f.write(content) 

        return out_path


class OpenAIBatchRunner:
    """
    Orchestrates the high-level workflow for running a Batch API experiment.
    """
    def __init__(self, generator: OpenAIBatchGenerator):
        """
        Initializes the runner with a specific generator.

        Args:
            generator (OpenAIBatchGenerator): The generator instance to use for API logic.
        """
        self.generator = generator

    def run_batch(self, df: pd.DataFrame, config, base_path: str = "outputs") -> str:
        """
        Executes the full pipeline: preparation, upload, execution, and download.

        Args:
            df (pd.DataFrame): The source DataFrame for prompts.
            config: Configuration object containing model params and .id() method.
            base_path (str): Root directory for output artifacts.

        Returns:
            str: Path to the final response file.
        """
        exp_id = config.id()

        input_path = f"{base_path}/{exp_id}_input.jsonl"
        output_path = f"{base_path}/{exp_id}_responses.jsonl"

        jsonl_path = self.generator.create_batch_file(
            df=df,
            out_path=input_path,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        file_id = self.generator.upload_file(jsonl_path)
        batch_id = self.generator.create_batch(file_id)
        batch = self.generator.wait_batch(batch_id)

        output_path = self.generator.download_results(batch, out_path=output_path)

        return output_path
    def recover_batch(self, batch_id: str, config, base_path="outputs"):
        """
        Recovers working batch if connection failed.
        """
        exp_id = config.id()
        output_path = f"{base_path}/{exp_id}_responses.jsonl"
        

        batch = self.generator.wait_batch(batch_id)
        return self.generator.download_results(batch, out_path=output_path)
        



