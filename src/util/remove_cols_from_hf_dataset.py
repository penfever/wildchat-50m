from datasets import Dataset, load_dataset, concatenate_datasets, DatasetDict

import argparse
import pandas as pd

from generate_model_responses_v5 import cols_to_remove, push_to_hub_fallback

conv_keys = [
    "content",
    # "content_token_ids",
    # "country",
    # "cumulative_logprob", 
    "finish_reason",
    "hashed_ip",
    # "header",
    # "language",
    # "redacted",
    "role",
    # "state",
    # "timestamp",
    "toxic",
    # "turn_identifier"
]

def process_list_dicts_column(example, column_name="conversation", target_column="conversation"):
    """
    Process a column containing lists of dictionaries using datasets.map()
    
    Args:
        example: Single example from the dataset
        column_name: Name of the column containing lists of dictionaries
        target_column: Name of the new column to store processed values
    
    Returns:
        Dict with processed values
    """
    processed_values = []
    entry = example[column_name]
    
    if isinstance(entry, list):
        for dict_item in entry:
            if isinstance(dict_item, dict):
                new_dict = {k : dict_item.get(k, "") for k in conv_keys}
                # Customize this part based on what you want to extract/transform
                processed_values.append(
                    new_dict
                )
    return {target_column: processed_values}

def main(args):
    ds = load_dataset(args.dataset_name, split="train")
    ds = ds.map(
        lambda x: x,
        remove_columns=cols_to_remove
    )
    ds = ds.map(
        lambda x: process_list_dicts_column(x),
    )
    if not push_to_hub_fallback(ds, args.dataset_name):
        raise ValueError(f"Failed to push to hub after 3 attempts: {args.dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for generate model responses')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    main(args)