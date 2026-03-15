import torch

class FederatedServer:
    """
    The Central Server in Federated Learning.
    Collect LoRA weights from clients and aggregate them using FedAvg.
    """
    def __init__(self):
        self.global_weights = None
        print("\n[Server] The central server started up")

    def aggregate_weights(self, client_weights_list):
        """
        Federated Averaging Algorithm
        Calculate the average of the weight matrices from the clients
        """
        print("\n[Server] Merging weights...")
        
        keys = client_weights_list[0].keys()
        averaged_weights = {}

        for key in keys:
            tensors = [weights[key] for weights in client_weights_list]
            stacked_tensors = torch.stack([t.to(torch.float32) for t in tensors])
            avg_tensor = torch.mean(stacked_tensors, dim=0)
            averaged_weights[key] = avg_tensor.to(tensors[0].dtype)
            
        self.global_weights = averaged_weights
        print("[Server] Done, new Global Model created")
        
        return self.global_weights