import numpy as np
from datasets import Dataset

class FederatedDataSplitter:
    """
    A utility class designed to partition a centralized dataset into multiple 
    smaller subsets for simulated Federated Learning (FL) clients.
    """
    def __init__(self, dataset: Dataset, num_clients: int = 2, seed: int = None):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        
    def split_iid(self):
        """
        Performs Independent and Identically Distributed (IID) data splitting.
        """
        import random
        if self.seed is None:
            shuffle_seed = random.randint(0, 1000000)
        else:
            shuffle_seed = self.seed
        
        shuffled_dataset = self.dataset.shuffle(seed=shuffle_seed)
        split_size = len(shuffled_dataset) // self.num_clients
        print(f"[DataSplitter] IID split with seed {shuffle_seed}")
        
        client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = len(shuffled_dataset) if i == self.num_clients - 1 else (i + 1) * split_size
            subset = shuffled_dataset.select(range(start_idx, end_idx))
            client_datasets.append(subset)
            
        return client_datasets

    def split_non_iid(self, alpha: float = 0.5):
        """
        Performs Non-IID data splitting 
        using Dirichlet distribution.
        """
        import random
        if self.seed is None:
            dirichlet_seed = random.randint(0, 1000000)
        else:
            dirichlet_seed = self.seed
            
        np.random.seed(dirichlet_seed)
        print(f"[DataSplitter] Non-IID split with seed {dirichlet_seed}")
        
        num_classes = 5
        answers = [str(ans).lower().strip() for ans in self.dataset['answer']]
        classes = np.array([hash(ans) % num_classes for ans in answers])
        class_indices = {c: np.where(classes == c)[0] for c in range(num_classes)}
        client_indices = [[] for _ in range(self.num_clients)]
        
        for c in range(num_classes):
            idx = class_indices[c]
            np.random.shuffle(idx)
            
            if len(idx) == 0:
                continue
                
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            
            idx_splits = np.split(idx, split_points)
            for i in range(self.num_clients):
                client_indices[i].extend(idx_splits[i].tolist())
                
        client_datasets = []
        for indices in client_indices:
            np.random.shuffle(indices)
            if len(indices) == 0:
                indices = [np.random.randint(0, len(self.dataset))]
                
            client_datasets.append(self.dataset.select(indices))
            
        print(f"[DataSplitter] Non-IID splitted (alpha={alpha}).")
        for i, ds in enumerate(client_datasets):
            print(f"Hospital {i+1} receives: {len(ds)} photos.")
            
        return client_datasets