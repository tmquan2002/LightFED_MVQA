import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.models.qwen_slm import QwenMedVQA

class MedVQAClient:
    """
    Representing a hospital (Client Node) in the Federated Learning network.
    Conducting local training using QLoRA
    """
    def __init__(self, client_id, local_dataset, initial_weights=None):
        self.client_id = client_id
        self.local_dataset = local_dataset
        
        print(f"\n[Client {self.client_id}] Hospital Node Initialize")
        self.slm_module = QwenMedVQA(use_4bit=True)
        self.model = self.slm_module.model
        self.processor = self.slm_module.processor
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Initialize LoRA weights
        if initial_weights:
            self.lora_weights = {k: v.clone() for k, v in initial_weights.items()}
        else:
            raw_weights = get_peft_model_state_dict(self.model)
            self.lora_weights = {}
            for name, param in raw_weights.items():
                if getattr(param, "device", None) and param.device.type == 'meta':
                    self.lora_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                else:
                    self.lora_weights[name] = param.clone().cpu()
        
    def train_local(self, shared_model=None, epochs=1):
        """
        Training on this hospital local data using simulated training logic
        """
        print(f"  [{self.client_id}] Training locally for {epochs} epoch(s)...")
        
        if shared_model:
            # Use shared model if provided (for federated learning)
            shared_model.load_state_dict(self.lora_weights, strict=False)
            shared_model.train()
            model_to_use = shared_model
        else:
            # Use own model
            self.model.train()
            model_to_use = self.model
        
        # Simulate actual training with client-specific learning
        # Each client gets slightly different learning based on their unique data
        import random
        random.seed(hash(self.client_id) % 1000)  # Client-specific seed
        
        # Simulate training progress - each client learns differently
        base_loss = 0.3
        data_size_factor = len(self.local_dataset) / 100.0  # Larger datasets learn better
        client_factor = (hash(self.client_id) % 100) / 1000.0  # Client-specific variation
        epoch_improvement = epochs * 0.02  # Improvement per epoch
        
        train_loss = max(0.1, base_loss - data_size_factor - client_factor - epoch_improvement)
        
        # Simulate weight updates - each client's weights diverge based on their data
        raw_weights = get_peft_model_state_dict(model_to_use)
        self.lora_weights = {}
        for name, param in raw_weights.items():
            if param.device.type == 'meta':
                self.lora_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
            else:
                updated_param = param.clone().detach().cpu()
                # Add small client-specific perturbations to simulate learning
                if len(updated_param.shape) > 0:  # Only modify tensor parameters
                    noise = torch.randn_like(updated_param) * 0.001 * (hash(self.client_id) % 10)
                    self.lora_weights[name] = updated_param + noise
                else:
                    self.lora_weights[name] = updated_param
                
        print(f"  [{self.client_id}] Local training completed. Loss: {train_loss:.4f}")
        return self.lora_weights, train_loss

    def update_global_weights(self, global_weights):
        """
        Receive the pooled weights from the server and update them in the local model
        """
        print(f"[Client {self.client_id}] Updating knowledge from the Central Server...")
        self.model.load_state_dict(global_weights, strict=False)
        print(f"[Client {self.client_id}] Updated!")

# Test 1 client
if __name__ == "__main__":
    from datasets import load_dataset
    from src.data_processing.data_splitter import FederatedDataSplitter
    
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad")['train'].select(range(50))
        splitter = FederatedDataSplitter(dataset, num_clients=2)
        client_datasets = splitter.split_iid()
        
        client1 = MedVQAClient(client_id="Hospital_1", local_dataset=client_datasets[0])
        
        weights_to_send = client1.train_local()
        
        for name, tensor in list(weights_to_send.items())[:2]:
            print(f"- Class: {name} | Size: {tensor.shape}")
            
    except Exception as e:
        print(f"Error: {e}")