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

def load_full_datasets(dataset_name, save_path=None):
    """
    Downloads the dataset and saves both 'train' and 'test' splits to disk.
    """
    print(f"[Loader] Loading full dataset: {dataset_name}...")
    full_dataset = load_dataset(dataset_name)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # Save each split separately (train, test, validation if any)
        for split in full_dataset.keys():
            split_path = os.path.join(save_path, split)
            full_dataset[split].save_to_disk(split_path)
            print(f"[Loader] Saved {dataset_name} ({split} split) at: {split_path}")
        
    return full_dataset

if __name__ == "__main__":
    print("--- Download and Save Full Datasets (Train + Test) ---")
    
    # 1. Process VQA-RAD
    # This will create ./data/vqa_rad_full/train and ./data/vqa_rad_full/test
    load_full_datasets(
        dataset_name="flaviagiammarino/vqa-rad", 
        save_path="./data/vqa_rad_full"
    )
    
    # 2. Process PathVQA
    # This will create ./data/path_vqa_full/train and ./data/path_vqa_full/test
    load_full_datasets(
        dataset_name="flaviagiammarino/path-vqa",
        save_path="./data/path_vqa_full"
    )