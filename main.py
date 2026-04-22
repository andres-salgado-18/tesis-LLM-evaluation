import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.models.config import (
    OpenAIGenerationConfig, 
    OpenAIEmbeddingConfig,
    all_MiniLM_L6_v2Config,
    all_mpnet_base_v2Config,
    paraphrase_multilingual_MiniLM_L12_v2Config
)
from src.experiments.run import run_experiments
from src.models.generative import OpenAIBatchGenerator, OpenAIBatchRunner
from src.pipeline.generative_pipeline import OpenAIBatchPipeline
from src.models.embeddings import OpenAIEmbedder, LocalEmbedder
from src.pipeline.embeddings_pipeline import EmbeddingsPipeline

def main():
    load_dotenv()
    client = OpenAI()

    parser = argparse.ArgumentParser(description="LLM Pipeline: Generation (API) and Embeddings (Hybrid)")
    
    parser.add_argument("--task", choices=["generate", "embed"], required=True, help="Task to execute")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    
    # NUEVO: Argumento para recuperar un batch fallido
    parser.add_argument("--batch_id", type=str, default=None, help="Optional: OpenAI Batch ID to recover an existing process")
    
    gen_group = parser.add_argument_group("Generation Options (OpenAI API)")
    gen_group.add_argument("--gen_model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    gen_group.add_argument("--temp", type=float, default=0.1, help="Temperature")
    gen_group.add_argument("--max_tokens", type=int, default=500, help="Max tokens")
    
    emb_group = parser.add_argument_group("Embeddings Options (API or Local)")
    emb_group.add_argument(
        "--emb_model", 
        choices=[
            "text-embedding-3-small", 
            "text-embedding-3-large", 
            "all-MiniLM-L6-v2", 
            "all-mpnet-base-v2", 
            "paraphrase-multilingual-MiniLM-L12-v2"
        ], 
        default="text-embedding-3-small"
    )
    emb_group.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    if args.task == "generate":
        df = pd.read_csv(args.input, dtype=str).reset_index()
        
        generator = OpenAIBatchGenerator(client)
        runner = OpenAIBatchRunner(generator)
        pipeline = OpenAIBatchPipeline(runner)
        
        config = OpenAIGenerationConfig(
            model=args.gen_model, 
            temperature=args.temp, 
            max_tokens=args.max_tokens
        )

        if args.batch_id:
            print(f"RECOVERING MODE: Connecting to batch {args.batch_id}")
            pipeline.run(df, config, existing_batch_id=args.batch_id)
        else:
            print(f"NEW RUN: Executing generation with {args.gen_model}")
            run_experiments(df, pipeline, [config])

    elif args.task == "embed":
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)

        if args.emb_model in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]:
            print(f"Using local embedding model: {args.emb_model}")
            embedder = LocalEmbedder(args.emb_model)
            
            mapping = {
                "all-MiniLM-L6-v2": all_MiniLM_L6_v2Config,
                "all-mpnet-base-v2": all_mpnet_base_v2Config,
                "paraphrase-multilingual-MiniLM-L12-v2": paraphrase_multilingual_MiniLM_L12_v2Config
            }
            config_class = mapping[args.emb_model]
        else:
            print(f"Executing embedding generation with OpenAI model: {args.emb_model}")
            embedder = OpenAIEmbedder(client, model=args.emb_model)
            config_class = OpenAIEmbeddingConfig
            
        pipeline = EmbeddingsPipeline(embedder)
        configs = [config_class(model=args.emb_model, batch_size=args.batch_size)]
        
        run_experiments(df, pipeline, configs)

    #todo: opción de análisis de embeddings, usando 'metrics'

    print("Process completed successfully.")

if __name__ == "__main__":
    main()