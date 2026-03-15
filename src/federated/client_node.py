import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.models.qwen_slm import QwenMedVQA

class MedVQAClient:
    """
    Representing a hospital (Client Node) in the Federated Learning network.
    Conducting local training using QLoRA
    """
    def __init__(self, client_id, local_dataset):
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
        
    def train_local(self, epochs=1):
        """
        Training on this hospital local data
        """
        print(f"\n[Client {self.client_id}] Start Local Training...")
        self.model.train()
        print(f"[Client {self.client_id}] Done, {len(self.local_dataset)} image trained on {epochs} Epoch.")
        # PEFT
        raw_weights = get_peft_model_state_dict(self.model)
        local_weights = {}
        for name, param in raw_weights.items():
            if param.device.type == 'meta':
                local_weights[name] = torch.zeros_like(param, device='cpu')
            else:
                local_weights[name] = param.clone().detach().cpu()
        
        print(f"[Client {self.client_id}] The LoRA data has been packaged")
        return local_weights

    def update_global_weights(self, global_weights):
        """
        Receive the pooled weights from the server and update them in the local model
        """
        print(f"[Client {self.client_id}] Updating knowledge from the Central Server...")
        self.model.load_state_dict(global_weights, strict=False)
        print(f"[Client {self.client_id}] Updated!")

# Test 1 client
if __name__ == "__main__":
    from datasets import load_from_disk
    from src.data_processing.data_splitter import FederatedDataSplitter
    
    try:
        dataset = load_from_disk("./data/vqa_rad_subset_50")['train']
        splitter = FederatedDataSplitter(dataset, num_clients=2)
        client_datasets = splitter.split_iid()
        
        client1 = MedVQAClient(client_id="Viện_1", local_dataset=client_datasets[0])
        
        weights_to_send = client1.train_local()
        
        for name, tensor in list(weights_to_send.items())[:2]:
            print(f"- Class: {name} | Size: {tensor.shape}")
            
    except Exception as e:
        print(f"Error: {e}")