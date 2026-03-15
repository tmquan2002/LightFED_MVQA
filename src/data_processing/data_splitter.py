import numpy as np
from datasets import Dataset

class FederatedDataSplitter:
    """
    A utility class designed to partition a centralized dataset into multiple 
    smaller subsets for simulated Federated Learning (FL) clients.
    """
    def __init__(self, dataset: Dataset, num_clients: int = 5):
        """
        Initializes the Federated Data Splitter.
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.total_samples = len(dataset)
        
    def split_iid(self):
        """
        Performs Independent and Identically Distributed (IID) data splitting.
        """
        print(f"[Splitter] Assigning IIDs to {self.num_clients} Clients...")
        indices = np.random.permutation(self.total_samples)
        split_indices = np.array_split(indices, self.num_clients)
        
        client_datasets = []
        for i, idx_array in enumerate(split_indices):
            client_subset = self.dataset.select(idx_array.tolist())
            client_datasets.append(client_subset)
            print(f"  -> Client {i+1} nhận {len(client_subset)} mẫu.")
            
        return client_datasets

    def split_non_iid_by_quantity(self, min_size=0.05, max_size=0.4):
        """
        Performs Non-IID (Non-Independent and Identically Distributed) data splitting 
        using Dirichlet distribution.
        """
        print(f"[Splitter] Allocating Non-IID (Quantity Difference) to {self.num_clients} Clients...")
        indices = np.random.permutation(self.total_samples)
        proportions = np.random.uniform(min_size, max_size, self.num_clients)
        proportions = proportions / proportions.sum() 
        sample_counts = (proportions * self.total_samples).astype(int)
        sample_counts[-1] = self.total_samples - sample_counts[:-1].sum()
        
        client_datasets = []
        current_idx = 0
        for i, count in enumerate(sample_counts):
            idx_array = indices[current_idx : current_idx + count]
            client_subset = self.dataset.select(idx_array.tolist())
            client_datasets.append(client_subset)
            current_idx += count
            
            print(f" Hospital (Client {i+1}) size {'Large' if count > self.total_samples/self.num_clients else 'Small'} receives {count} samples ({proportions[i]*100:.1f}%).")
            
        return client_datasets

# Testing module
if __name__ == "__main__":
    from datasets import load_from_disk
    
    try:
        # Load VQA-RAD subset
        dataset = load_from_disk("../../data/vqa_rad_subset_50")['train']
        splitter = FederatedDataSplitter(dataset, num_clients=3)
        print("--- Test IID ---")
        iid_clients = splitter.split_iid()
        
        print("\n--- Test NON-IID ---")
        non_iid_clients = splitter.split_non_iid_by_quantity()
        
    except FileNotFoundError:
        print("Data not found. Please run file data_subset_generator.py first.")