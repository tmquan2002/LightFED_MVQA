import torch

class FederatedServer:
    """
    The Central Server in Federated Learning.
    Collect LoRA weights from clients and aggregate them using FedAvg.
    """
    def __init__(self):
        self.global_weights = None
        print("\n[Server] The central server started up")

    def aggregate_weights(self, client_weights_list, client_sizes=None):
        """
        Federated Averaging Algorithm
        Calculate the average of the weight matrices from the clients.
        Supports optional dataset-weighted aggregation.
        """
        print("\n[Server] Merging weights...")
        
        if client_sizes is not None:
            total_samples = sum(client_sizes)
            weights = [size / total_samples for size in client_sizes]
        else:
            weights = [1.0 / len(client_weights_list)] * len(client_weights_list)
        
        keys = client_weights_list[0].keys()
        averaged_weights = {}

        for key in keys:
            tensors = [w[key] for w in client_weights_list]
            # Weighted average aggregation
            weighted_tensors = [t.to(torch.float32) * w for t, w in zip(tensors, weights)]
            avg_tensor = torch.stack(weighted_tensors).sum(dim=0)
            averaged_weights[key] = avg_tensor.to(tensors[0].dtype)
            
        self.global_weights = averaged_weights
        print("[Server] Done, new Global Model created")
        
        return self.global_weights