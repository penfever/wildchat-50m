import argparse
from datasets import load_dataset, load_from_disk
import random

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='View random samples from a HuggingFace dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path or name of the HuggingFace dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to load (default: train)')
    parser.add_argument('--n', type=int, default=10, help='Number of random samples to show (default: 10)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        try:
            print(f"Loading dataset: {args.dataset_path}")
            dataset = load_from_disk(args.dataset_path)
        except:
            # Load the dataset
            print(f"Loading dataset: {args.dataset_path}")
            dataset = load_dataset(args.dataset_path, split=args.split)
        
        # Get total size
        total_size = len(dataset)
        print(f"Dataset size: {total_size} examples")
        
        # Generate random indices
        sample_size = min(args.n, total_size)
        indices = random.sample(range(total_size), sample_size)
        
        # Print random samples
        print(f"\nShowing {sample_size} random examples:\n")
        for i, idx in enumerate(indices, 1):
            print(f"=== Sample {i} (Index: {idx}) ===")
            print(dataset[idx])
            print()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())