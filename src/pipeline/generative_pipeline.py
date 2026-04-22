from src.utils.io_handler import load_responses
from pathlib import Path

class OpenAIBatchPipeline:
    """
    Orchestrates the end-to-end generative AI workflow using the OpenAI Batch API.
    
    This pipeline is designed for resilience, supporting both new batch executions 
    and recovery of existing batches in case of network or process failure.
    """
    def __init__(self, runner):
        """
        Initializes the pipeline with a batch runner.

        Args:
            runner (OpenAIBatchRunner): The runner instance responsible for 
                                        managing the Batch API lifecycle.
        """
        self.runner = runner

    def run(self, df, config, base_path="outputs", existing_batch_id=None):
        """
        Executes the generative pipeline by either starting a new batch or recovering an existing one.

        Args:
            df (pd.DataFrame): Input DataFrame containing the prompts and metadata.
            config: Configuration object containing model settings and the .id() method.
            base_path (str): Directory for storing input/output JSONL files and the final Parquet.
            existing_batch_id (str, optional): If provided, the pipeline skips the upload and 
                                               creation steps, moving directly to monitoring 
                                               and downloading the specified batch.

        Returns:
            tuple: (pd.DataFrame, str) The DataFrame aligned with generated responses 
                   and the final output file path.
        """
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        
        if existing_batch_id:
            print(f"Re-engaging with existing batch: {existing_batch_id}")
            output_path = self.runner.recover_batch(existing_batch_id, config, base_path)
        else:
            
            output_path = self.runner.run_batch(df, config, base_path)

        responses = load_responses(output_path)
        
        df_out = df.copy()
        df_out["response"] = df_out["index"].apply(lambda x: f"idx_{x}").map(responses)

        out_path = f"{base_path}/{config.id()}.parquet"
        df_out.to_parquet(out_path, index=False)

        return df_out, out_path