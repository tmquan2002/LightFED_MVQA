import os
import random
from datasets import load_dataset, DatasetDict

def create_subset(dataset_name, num_samples=50, save_path=None):
    """
    Load the original dataset and create a small subset, saving it to your hard drive.
    """
    print(f"[Loader] Loading dataset: {dataset_name}...")
    full_dataset = load_dataset(dataset_name)
    
    total_train = len(full_dataset['train'])
    num_train = min(num_samples, total_train)
    
    # Random data
    random_indices = random.sample(range(total_train), num_train)
    subset_train = full_dataset['train'].select(random_indices)
    
    subset_dict = DatasetDict({'train': subset_train})
    
    # Save to hard drive
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        subset_dict.save_to_disk(save_path)
        print(f"[Loader] Saved {num_train} samples at: {save_path}\n")
        
    return subset_dict

if __name__ == "__main__":
    print("--- Create Subset ---")
    
    # Create subset for VQA-RAD
    create_subset(
        dataset_name="flaviagiammarino/vqa-rad", 
        num_samples=50, 
        save_path="./data/vqa_rad_subset_50"
    )
    
    # Create subset for PathVQA
    create_subset(
        dataset_name="flaviagiammarino/path-vqa", 
        num_samples=100, 
        save_path="./data/path_vqa_subset_100"
    )