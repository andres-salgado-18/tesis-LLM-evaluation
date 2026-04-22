import pandas as pd
import json
import numpy as np

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads an experimental dataset from a CSV file with predefined string types.
    
    Args:
        filepath (str): Path to the source CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the experimental prompts and metadata.
    """
    dtype_dict = {
        "scenario": str,
        "prompt": str,
        "entity_familiarity": str,
        "context_coherence": str,
        "ethical_pressure": str,
        "prompt_structure": str,
        "lang": str
    }
    return pd.read_csv(filepath, dtype=dtype_dict)

def save_json_results(data: list[dict], filename: str):
    """
    Serializes a list of dictionaries to a JSON file with pretty printing.
    
    Args:
        data (list[dict]): The results data to save.
        filename (str): The destination file path.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        

def load_embeddings_from_json(path):
    """
    Extracts numerical embeddings and raw metadata from a JSON results file.
    
    Args:
        path (str): Path to the JSON file containing embedding data.
        
    Returns:
        tuple: A tuple containing (numpy_matrix_of_embeddings, original_json_data).
    """
    with open(path, "r") as f:
        data = json.load(f)

    embeddings = np.array([row["embedding_output"] for row in data])
    return embeddings, data


def load_responses(path: str):
    """
    Parses an OpenAI Batch API results file (.jsonl) and extracts generated text.
    
    This function handles API errors, batch failures, and malformed responses 
    by returning descriptive error strings instead of crashing.
    
    Args:
        path (str): Path to the .jsonl results file.
        
    Returns:
        dict[str, str]: A mapping of custom_id (e.g., 'idx_0') to the response text.
    """
    responses = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            cid = obj["custom_id"]
            text = obj["response"]["body"]["choices"][0]["message"]["content"]

            responses[cid] = text

    return responses

def load_responses(path):
    responses = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            cid = obj.get("custom_id")

            if obj.get("error") is not None:
                responses[cid] = f"BATCH_ERROR: {obj['error']}"
                continue

            response_data = obj.get("response", {})
            status_code = response_data.get("status_code")

            if status_code == 200:
                try:
                    text = response_data["body"]["choices"][0]["message"]["content"]
                    responses[cid] = text
                except (KeyError, TypeError):
                    responses[cid] = "ERROR: estructura inesperada"
            else:
                error_info = response_data.get("body", {}).get("error", "Unknown error")
                responses[cid] = f"API_ERROR_{status_code}: {error_info}"

    return responses