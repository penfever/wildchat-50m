import argparse
from itertools import islice
from pprint import pprint
from typing import Dict, Any, Optional, Tuple, List
import logging
import traceback
import time

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets, DatasetDict
from vllm import LLM, SamplingParams
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoTokenizer
import numpy as np

from judgment import get_judge_template
from utils import convert_dataset_types

cols_to_remove = [
    "timestamp",
    "turn",
    "language",
    "openai_moderation",
    "detoxify_moderation",
    "toxic",
    "redacted",
    "state",
    "country",
    "hashed_ip",
    "header",
]

def push_to_hub_fallback(response_dataset, repo_id, max_retries=3, delay=60):
    """
    Push dataset to HuggingFace Hub with retry logic
    
    Args:
        response_dataset: Dataset to push
        repo_id: Full repository ID
        max_retries: Maximum number of retries per attempt
        delay: Delay in seconds between retries
    """
    for attempt in range(max_retries):
        try:
            response_dataset.push_to_hub(repo_id=repo_id, private=True)
            print(f"Successfully pushed to {repo_id}")
            return True
        except HfHubHTTPError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e.response.status_code} - {e.response.text}")
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
    return False

def batch_iterator(iterable, batch_size):
    """Create batches from an iterable"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def get_features(data):
    if isinstance(data, Dataset):
        return data.features
    elif isinstance(data, DatasetDict):
        # Get features from first available split
        first_split = next(iter(data.values()))
        return first_split.features
    else:
        raise TypeError(f"Expected Dataset or DatasetDict, got {type(data)}")

# Alternative approach using batch processing
def update_nested_field_batch(dataset: Dataset, new_values: dict, default_value_for_logprob: str, args) -> Dataset:
    """
    Update nested fields using batch processing for better performance.
    
    Args:
        dataset: Input dataset
        new_values: List of new content values to insert
        
    Returns:
        Dataset with updated nested fields
    """


    def update_batch(examples: dict, reference_column="conversation") -> dict:
        # Create new conversations list with explicit schema
        updated_conversations = []
        
        for conv in examples[reference_column]:
            new_conv = []
            len_conv = len(conv)
            
            for j in range(0, len_conv, 2):
                # Copy human message as-is
                new_conv.append(dict(conv[j]))
                
                # Update assistant message with new values
                if j + 1 < len_conv:
                    key = tuple(conv[j]["content_token_ids"])
                    assistant_msg = dict(conv[j+1])  # Create copy of original message
                    
                    if key in new_values:
                        assistant_msg.update({
                            'content': str(new_values[key][0]),
                            'cumulative_logprob': default_value_for_logprob,
                            'finish_reason': str(new_values[key][2])
                        })
                    else:
                        assistant_msg.update({
                            'content': "",
                            'cumulative_logprob': default_value_for_logprob,
                            'finish_reason': "no response"
                        })
                    new_conv.append(assistant_msg)
            
            updated_conversations.append(new_conv)
        return {reference_column: updated_conversations}

    def update_batch_with_judgments(examples: dict, reference_column="conversation", target_field="field") -> dict:
        # Create new conversations list with explicit schema
        updated_conversations = []
        
        try:
            for i, conv in enumerate(examples[reference_column]):
                new_conv = []
                len_conv = len(conv)
                
                for j in range(0, len_conv, 2):
                    # Copy human message as-is
                    new_conv.append(dict(conv[j]))
                    
                    # Update assistant message with new values
                    if j + 1 < len_conv:
                        key = " ".join(str(i) for i in conv[j]["content_token_ids"])
                        assistant_msg = dict(conv[j+1])  # Create copy of original message
                        
                        if key in new_values:
                            assistant_msg.update({
                                target_field + '_content': str(new_values[key][0]),
                                target_field + '_cumulative_logprob': str(new_values[key][1]),
                                target_field + '_logprob': str(new_values[key][2])
                            })
                        else:
                            assistant_msg.update({
                                target_field + '_content': "",
                                target_field + '_cumulative_logprob': str(default_value_for_logprob),
                                target_field + '_logprob': ""
                            })
                            
                        new_conv.append(assistant_msg)
                updated_conversations.append(new_conv)
            return {reference_column: updated_conversations}
        except Exception as e:
            print(f"Error in update_batch_with_judgments: {str(e)}")
            traceback.print_exc()
            print(f"Examples structure: {type(examples)}")
            print(f"Reference column: {reference_column}")
            print(f"Target field: {target_field}")
            raise

    if args.gen_judgments:
        new_field_name = f"judgment_{args.model_name.replace('/', '_')}_{args.reference_column}_{args.judgment_factor}"
        updated_dataset = dataset.map(
            update_batch_with_judgments,
            batched=True,
            with_indices=False,
            batch_size=100,  # Adjust based on your needs
            desc="Updating conversation content in batches",
            fn_kwargs={'reference_column': args.reference_column, 'target_field': new_field_name},
        )
    else:
        updated_dataset = dataset.map(
            update_batch,
            batched=True,
            with_indices=False,
            batch_size=100,  # Adjust based on your needs
            desc="Updating conversation content in batches",
            fn_kwargs={'reference_column': args.reference_column},
        )
        # Then filter out empty conversations
    return updated_dataset.filter(
        lambda x: len(x[args.reference_column]) > 0,
        desc="Filtering out empty conversations"
    )
    # except Exception as e:
    #     print("Exception in update_nested_field_batch:")
    #     print(e, "\n")
    #     return []

def truncate_content(p, threshold, tokenizer):
    p_tok = tokenizer.encode(p)
    if len(p_tok) > threshold:
        p_tok = p_tok[:threshold - 1] + [tokenizer.eos_token_id]
    return tokenizer.decode(p_tok)

def merge_datasets_by_hash(source_dataset, target_dataset):
    """
    Merge two HF datasets by replacing rows in target dataset with matching rows from source dataset,
    using conversation_hash as the primary key.
    
    Args:
        source_dataset: HuggingFace dataset containing the source data
        target_dataset: HuggingFace dataset to be updated with source data
        
    Returns:
        Updated dataset with merged data
    """
    # Get set of conversation hashes from source dataset
    source_hashes = set(source_dataset['train']['conversation_hash'])
    
    # Create a mapping of hash to row for quick lookup
    hash_to_row = {row['conversation_hash']: row for row in source_dataset['train']}
    
    # Function to replace matching rows
    def update_row(row):
        if row['conversation_hash'] in source_hashes:
            return hash_to_row[row['conversation_hash']]
        return row
    
    # Apply the update to target dataset
    updated_dataset = target_dataset['train'].map(update_row)
    
    return updated_dataset

def convert_to_float_array(logprobs_list: List[Dict[str, Any]]) -> np.ndarray:
    """
    Convert VLLM logprobs to a fixed-size array of floats, specifically looking for
    numeric values after the sequence "Score: ".
    
    Args:
        logprobs_list: List of logprobs objects from VLLM output
        
    Returns:
        np.ndarray: Fixed-size array of length 10 containing probabilities
    """
    result = np.zeros(10, dtype=np.float32)
    
    if not logprobs_list:
        return result
        
    # Find the position after "Score: " sequence
    score_pos = -1
    for i, token_probs in enumerate(logprobs_list):
        top_token = max(token_probs.items(), key=lambda x: x[1].logprob)
        if top_token[1].decoded_token == "Score":
            # Check if next tokens are ":" and " "
            if (i + 2 < len(logprobs_list) and
                any(v.decoded_token == ":" for v in logprobs_list[i + 1].values()) and
                any(v.decoded_token == " " for v in logprobs_list[i + 2].values())):
                score_pos = i + 3
                break
    
    if score_pos == -1 or score_pos >= len(logprobs_list):
        return result
        
    # Process numeric tokens after "Score: "
    for token_id, logprob_obj in logprobs_list[score_pos].items():
        try:
            token = logprob_obj.decoded_token.strip()
            if token.isdigit():
                value = int(token)
                if 1 <= value <= 10:
                    result[value - 1] = np.exp(float(logprob_obj.logprob))  # Convert logprob to probability
        except (ValueError, AttributeError) as e:
            continue
            
    return result

def process_outputs(outputs, default_value_for_logprob=0.0):
    """Process outputs to ensure consistent types for HF datasets"""
    output_texts = []
    
    for output in outputs:
        try:
            # Get the first output
            first_output = output.outputs[0]
            
            # Extract and convert text
            text = str(getattr(first_output, 'text', ""))
            
            # Extract and convert logprob
            logprob = str(round(getattr(first_output, 'cumulative_logprob', default_value_for_logprob), 3) or 0.0)
            
            # Extract and convert logprobs array
            logprobs_array = convert_to_float_array(getattr(first_output, 'logprobs', []))
            
            # Create tuple with verified types
            # output_tuple = (text, logprob, 0.0)
            output_tuple = (text, logprob, logprobs_array)
            output_texts.append(output_tuple)
            
        except Exception as e:
            print(f"Error processing output: {e}")
            # Add a default tuple in case of error
            # output_texts.append(("", 0.0, 0.0))
            output_texts.append(("", 0.0, np.zeros(10, dtype=np.float32)))
    
    return output_texts

def process_batch(batch, model_name, tokenizer, llm, sampling_params, debug=False, truncate_threshold=-1, logprob_dtype="null", system_prompt="You are a helpful assistant.", args=None):
    """Process a batch of conversations"""
    # Count number of turns in the conversation

    # Prepare all prompts in the batch
    prompt_token_ids = []
    for row in batch:
        len_conv = len(row[args.reference_column])
        if len_conv % 2 != 0:
            # Skip odd-length conversations
            continue
        if truncate_threshold > 0:
            threshold_per_turn = max(256, int(0.95 * truncate_threshold // len_conv))
            for i in range(0, len_conv):
                row[args.reference_column][i]['content'] = str(truncate_content(row[args.reference_column][i]['content'], threshold_per_turn, tokenizer))
        for i in range(0, len_conv, 2):
            if args.gen_judgments:
                conversation_until_now = [{"role": "system", "content": system_prompt}] + row[args.reference_column][:i+1]
                conversation_until_now[-1]["content"] = conversation_until_now[-1]["content"] + " <|END_OF_CONVERSATION|>"
            else:
                conversation_until_now = row[args.reference_column][:i+1]
            tokenized_conv_until_now = tokenizer.apply_chat_template(conversation_until_now, add_generation_prompt=True)
            if truncate_threshold > 0 and len(tokenized_conv_until_now) > truncate_threshold:
                tokenized_conv_until_now = tokenized_conv_until_now[:truncate_threshold - 256] + [tokenizer.eos_token_id]
            row[args.reference_column][i]['content_token_ids'] = tokenized_conv_until_now
            prompt_token_ids.append(tokenized_conv_until_now)
            
    # Generate responses for all prompts in the batch
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    if logprob_dtype == 'null':
        default_value_for_logprob = None
    else:
        default_value_for_logprob = 0.0

    if args.gen_judgments:
        output_texts = process_outputs(outputs, default_value_for_logprob)
        tokenid_to_text = {" ".join(str(i) for i in prompt_token_ids[i]) : output_texts[i] for i in range(len(prompt_token_ids))}
    else:
        # Extract output texts using list comprehension
        output_texts = [(
            str(getattr(output.outputs[0], 'text', "")),  # Handle case where logprob isn't available
            getattr(output.outputs[0], 'cumulative_logprob', default_value_for_logprob),  # Handle case where logprob isn't available
            str(getattr(output.outputs[0], 'finish_reason', ""))  # Handle case where finish_reason isn't available
        ) for output in outputs]
        tokenid_to_text = {tuple(prompt_token_ids[i]): output_texts[i] for i in range(len(prompt_token_ids))}
    try:
        assert len(prompt_token_ids) == len(output_texts)
    except AssertionError:
        print("Error: Number of conversations and output texts do not match")
        return []

    # Reformat batch
    if isinstance(batch, list):
        batch = Dataset.from_list(batch)
    # Update model name
    # batch = batch.remove_columns(['model']).add_column('model', [model_name] * len(batch))

    batch = update_nested_field_batch(batch, tokenid_to_text, default_value_for_logprob, args)
       
    return batch

def main(args):
    
    # Construct full save path
    if args.gen_judgments:
        model_save_path = f"{args.dataset_name.replace('/', '_')}-jdgfct-{args.judgment_factor}"
        model_save_path = model_save_path.replace("penfever_allenai_WildChat-1M-Full-", "")
    else:
        model_save_path = f"{args.dataset_name.replace('/', '_')}-{args.model_name.replace('/', '_')}"
    model_save_path = model_save_path.replace(" ", "")
    if args.seed != 0:
        model_save_path = model_save_path + f"s{args.seed}"
    update_in_place=False

    # Load prompting dataset
    print("Loading prompting dataset \n")
    ds = load_dataset(args.dataset_name)
    print(f"Loaded {len(ds['train'])} conversations. \n")
    if args.slim_dataset:
        ds = ds.map(
            lambda x: x,
            remove_columns=cols_to_remove
        )

    # If target dataset is not empty, load it
    load_success = False
    if not args.overwrite:
        try:
            print("Loading existing processed dataset \n")
            response_dataset = load_dataset(f"penfever/{model_save_path}", keep_in_memory=True)
            if args.slim_dataset:
                response_dataset = response_dataset.map(
                    lambda x: x,
                    remove_columns=cols_to_remove
                )
            
            # Get set of conversation hashes already processed
            processed_hashes = set(response_dataset['train']['conversation_hash'])

            # Filter the source dataset to only include unprocessed conversations
            ds_filtered = ds.filter(
                lambda example: example['conversation_hash'] not in processed_hashes,
                desc="Filtering out previously processed conversations"
            )

            if len(ds_filtered['train']) < 5000 and args.gen_judgments:
                print("Detected complete dataset during gen_judgments. Conversations will be updated in-place.")
                update_in_place=True
            else:
                ds = ds_filtered  # Always update the main dataset with the filtered version
            print(f"Done. Loaded {len(ds['train'])} unprocessed conversations. \n")
            load_success = True
        except Exception as e:
            print(f"Error while loading existing processed dataset: {e} \n")
            response_dataset = None
    else:
        response_dataset = None

    if load_success:
        features = get_features(response_dataset)
        conversation_features = features[args.reference_column]
        logprob_dtype = conversation_features[0]['cumulative_logprob'].dtype
    else:
        logprob_dtype = "null"
    # parameters
    if args.enforce_eager:
        enforce_eager_setting = True
    else:
        enforce_eager_setting = False
    max_output_len = round(0.2 * args.max_model_len)
    max_input_len = round(0.75 * args.max_model_len)

    # Initialize LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if args.gen_judgments:
        sampling_params = SamplingParams(max_tokens=10, temperature=0.2, logprobs=10, seed=args.seed)
    else:
        sampling_params = SamplingParams(max_tokens=max_output_len, seed=args.seed)
    llm = LLM(model=args.model_name, 
              max_model_len=args.max_model_len, 
              trust_remote_code=True,
              max_num_seqs=min(args.batch_size * 2, int(args.max_model_len / 1.5)),
              swap_space=10,
              enforce_eager=enforce_eager_setting,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
            )

    # Process data in batches
    total_processed = 0
    
    # Create batches from the dataset
    batches = batch_iterator(ds['train'], args.batch_size)
    
    if args.gen_judgments:
        system_prompt = get_judge_template(args.judgment_factor)
    else:
        system_prompt = "You are a helpful assistant."

    # Process each batch with a progress bar
    for batch in tqdm(batches, desc="Processing batches"):
        # Process the current batch
        processed_rows = process_batch(
            batch,
            args.model_name,
            tokenizer,
            llm,
            sampling_params,
            args.debug and total_processed % (10 * args.batch_size) < args.batch_size,
            max_input_len,
            logprob_dtype,
            system_prompt,
            args,
        )

        if len(processed_rows) == 0:
            continue

        if args.debug:
            print("Batch Sample\n\n" + "-" * 80)
            last_row_processed = processed_rows[-1]
            print("\nPrompt:")
            print(last_row_processed[args.reference_column][0]['content'])
            print("\nResponse:")
            print(last_row_processed[args.reference_column][1]['content'])
            print("-" * 80)

        if response_dataset is None:
            response_dataset = processed_rows
        elif update_in_place:
            try:
                try:
                    response_dataset['train'] = merge_datasets_by_hash(processed_rows, response_dataset['train'])
                except ValueError:
                    processed_rows = convert_dataset_types(processed_rows)
                    response_dataset['train'] = convert_dataset_types(response_dataset['train'])
                    response_dataset['train'] = merge_datasets_by_hash(processed_rows, response_dataset['train'])
            except:
                try:
                    response_dataset = merge_datasets_by_hash(processed_rows, response_dataset)
                except ValueError:
                    processed_rows = convert_dataset_types(processed_rows)
                    response_dataset = convert_dataset_types(response_dataset)
                    response_dataset = merge_datasets_by_hash(processed_rows, response_dataset)
        else:
            try:
                try:
                    response_dataset['train'] = concatenate_datasets([response_dataset['train'], processed_rows])
                except ValueError:
                    processed_rows = convert_dataset_types(processed_rows)
                    response_dataset['train'] = convert_dataset_types(response_dataset['train'])
                    response_dataset['train'] = concatenate_datasets([response_dataset['train'], processed_rows])
                print(f"Length of concatenated dataset: {len(response_dataset['train'])}")
            except:
                try:
                    response_dataset = concatenate_datasets([response_dataset, processed_rows])
                except ValueError:
                    processed_rows = convert_dataset_types(processed_rows)
                    response_dataset = convert_dataset_types(response_dataset)
                    response_dataset = concatenate_datasets([response_dataset, processed_rows])
                print(f"Length of concatenated dataset: {len(response_dataset)}")
        if args.debug:
            print("Concatenated Dataset (sample): \n\n" + "-" * 80)
            sample_ds = response_dataset.shuffle().select(range(3))
            print(sample_ds)
            print("-" * 80)
            pprint(sample_ds[0])
            print("-" * 80)
            pprint(sample_ds[1])
            print("-" * 80)
            pprint(sample_ds[2])
            print("-" * 80)
        
        # Update counter and save checkpoint if needed
        total_processed += 1
        if (args.max_batches_to_process > 0 and total_processed >= args.max_batches_to_process) \
            or (total_processed % 10 == 0):
            # Usage
            repo_id = f"penfever/{model_save_path}"
            if not push_to_hub_fallback(response_dataset, repo_id):
                raise ValueError(f"Failed to push to hub after 3 attempts: {repo_id}")

    
    # Final save
    repo_id = f"penfever/{model_save_path}"
    if not push_to_hub_fallback(response_dataset, repo_id):
        raise ValueError(f"Failed to push to hub after 3 attempts: {repo_id}")
    print(f"\nFinished processing {total_processed * args.batch_size} total rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for generate model responses')
    parser.add_argument('--model_name', type=str, required=True, 
                      help='Model name to generate responses (use HF format)')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./data',
                      help='Directory to save the processed dataset')
    parser.add_argument('--batch_size', type=int, default=5000, 
                      help='Batch size for generation')
    parser.add_argument('--max_batches_to_process', type=int, default=-1)
    parser.add_argument('--seed', type=int, help='Random seed', default=0)
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--debug', action='store_true', 
                      help='Enable debug mode to print prompts and responses every 10 iterations')
    parser.add_argument('--enforce_eager', action='store_true')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gen_judgments', action='store_true')
    parser.add_argument('--reference_column', type=str, default="conversation")
    parser.add_argument('--judgment_factor', type=str, default="Factuality")
    parser.add_argument('--slim_dataset', action='store_true')
    args = parser.parse_args()
    main(args)