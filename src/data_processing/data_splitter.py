import numpy as np
from datasets import Dataset

class FederatedDataSplitter:
    """
    A utility class designed to partition a centralized dataset into multiple 
    smaller subsets for simulated Federated Learning (FL) clients.
    """
    @staticmethod
    def is_closed_ended(answer) -> bool:
        """
        Determines if an answer is closed-ended (yes/no or short answers <= 2 words).
        """
        ans = str(answer).lower().strip()
        return ans in ['yes', 'no'] or len(ans.split()) <= 2

    def __init__(self, dataset: Dataset, num_clients: int = 2, seed: int = None, question_type: str = "all", max_samples: int = None):
        self.num_clients = num_clients
        self.seed = seed
        
        # 1. Filter by question type if requested
        filtered_dataset = dataset
        if question_type in ["closed", "open"]:
            indices = []
            for idx, item in enumerate(dataset):
                is_closed = self.is_closed_ended(item['answer'])
                
                if question_type == "closed" and is_closed:
                    indices.append(idx)
                elif question_type == "open" and not is_closed:
                    indices.append(idx)
            
            filtered_dataset = dataset.select(indices)
            print(f"[DataSplitter] Filtered dataset to {question_type} questions: {len(dataset)} -> {len(filtered_dataset)} samples.")
            
        # 2. Limit the number of samples if requested
        if max_samples is not None and len(filtered_dataset) > max_samples:
            import random
            if seed is not None:
                random.seed(seed)
            else:
                random.seed(42)  # default seed for stable subsetting
            
            indices = random.sample(range(len(filtered_dataset)), max_samples)
            # Sort indices to maintain order
            indices.sort()
            filtered_dataset = filtered_dataset.select(indices)
            print(f"[DataSplitter] Limited dataset to {max_samples} samples.")
            
        self.dataset = filtered_dataset
        
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

    def split_by_question_type(self):
        """
        Partitions the dataset such that a subset of clients receive only closed-ended
        questions, and the remaining clients receive only open-ended questions.
        """
        import random
        if self.seed is not None:
            random.seed(self.seed)
            
        closed_indices = []
        open_indices = []
        
        for idx, item in enumerate(self.dataset):
            if self.is_closed_ended(item['answer']):
                closed_indices.append(idx)
            else:
                open_indices.append(idx)
                
        # Split clients into two groups: first half closed-ended, second half open-ended
        num_closed_clients = self.num_clients // 2
        num_open_clients = self.num_clients - num_closed_clients
        
        print(f"[DataSplitter] Splitting by question type across clients:")
        print(f"  - {num_closed_clients} clients get closed-ended ({len(closed_indices)} total samples)")
        print(f"  - {num_open_clients} clients get open-ended ({len(open_indices)} total samples)")
        
        # Shuffle indices
        random.shuffle(closed_indices)
        random.shuffle(open_indices)
        
        client_datasets = [None] * self.num_clients
        
        # Distribute closed-ended
        if num_closed_clients > 0:
            closed_split_size = len(closed_indices) // num_closed_clients
            for i in range(num_closed_clients):
                start_idx = i * closed_split_size
                end_idx = len(closed_indices) if i == num_closed_clients - 1 else (i + 1) * closed_split_size
                client_datasets[i] = self.dataset.select(closed_indices[start_idx:end_idx])
                
        # Distribute open-ended
        if num_open_clients > 0:
            open_split_size = len(open_indices) // num_open_clients
            for i in range(num_open_clients):
                client_idx = num_closed_clients + i
                start_idx = i * open_split_size
                end_idx = len(open_indices) if i == num_open_clients - 1 else (i + 1) * open_split_size
                client_datasets[client_idx] = self.dataset.select(open_indices[start_idx:end_idx])
                
        for i, ds in enumerate(client_datasets):
            print(f"Hospital {i+1} receives: {len(ds)} photos.")
            
        return client_datasets