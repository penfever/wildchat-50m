from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Optional, Set, List, Dict

def get_first_message_content(conversation: List[Dict]) -> Optional[str]:
    """Extract content from the first message in a conversation.
    
    Args:
        conversation: List of message dictionaries
        
    Returns:
        Content string if available, None otherwise
    """
    if conversation and len(conversation) > 0:
        return conversation[0].get('content')
    return None

def is_unique(example: Dict, seen_contents: Set[str]) -> bool:
    """Check if the first message in conversation is unique.
    
    Args:
        example: Dataset example containing conversation
        seen_contents: Set of previously seen message contents
        
    Returns:
        bool: True if message is unique, False otherwise
    """
    content = get_first_message_content(example['conversation'])
    if content is None or content in seen_contents:
        return False
    seen_contents.add(content)
    return True

def limit_conversation_lengths(
    model_a: str,
    model_b: str,
    base_path: str = "penfever"
) -> None:
    """Limit conversations in dataset A to be no longer than corresponding ones in dataset B.
    
    Args:
        model_a: Name of the first model's dataset
        model_b: Name of the second model's dataset
        base_path: Base path for the datasets on HuggingFace
    """
    # Load datasets
    ds1 = load_dataset(f"{base_path}/{model_a}", split='train')
    ds2 = load_dataset(f"{base_path}/{model_b}", split='train')

    # Filter unique conversations
    seen_contents: Set[str] = set()
    ds1 = ds1.filter(lambda x: is_unique(x, seen_contents))

    # Find common conversation hashes
    indices = set(ds1['conversation_hash']).intersection(ds2['conversation_hash'])

    # Convert to pandas for easier manipulation
    selected_ds1 = ds1.to_pandas()[
        ds1.to_pandas()['conversation_hash'].isin(indices)
    ].sort_values('conversation_hash')
    
    selected_ds2 = ds2.to_pandas()[
        ds2.to_pandas()['conversation_hash'].isin(indices)
    ].sort_values('conversation_hash')

    # Clear memory
    del ds1, ds2

    print("Processing conversations...")
    
    # Process conversations
    for ((_, row1), (_, row2)) in tqdm(
        zip(selected_ds1.iterrows(), selected_ds2.iterrows()), 
        total=len(selected_ds1)
    ):
        conv1 = row1['conversation']
        conv2 = row2['conversation']
        
        # Skip if conversations are not aligned
        if len(conv1) != len(conv2) or len(conv1) % 2 != 0:
            continue
            
        # Limit content length
        for i in range(0, len(conv1), 2):
            conv1[i]["content"] = conv1[i]["content"][:len(conv2[i]["content"])]
        
        selected_ds1.at[row1.name, 'conversation'] = conv1

    # Convert back to HF dataset and push
    output_dataset = Dataset.from_pandas(selected_ds1, split='train')
    output_dataset.push_to_hub(f"{base_path}/{model_a}-lc")
    
    print("Dataset processing complete.")

if __name__ == "__main__":
    MODEL_A = "allenai_WildChat-1M-Full-Qwen_Qwen2.5-72B-Instruct"
    MODEL_B = "allenai_WildChat-1M-Full-meta-llama_Llama-3.3-70B-Instruct"
    
    limit_conversation_lengths(MODEL_A, MODEL_B)