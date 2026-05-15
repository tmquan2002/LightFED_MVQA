from numpy import random
import json
import os
import time
import torch
import psutil
import threading
from datetime import datetime
from datasets import load_from_disk, concatenate_datasets
from src.data_processing.data_splitter import FederatedDataSplitter
from src.federated.server import FederatedServer
from src.evaluation.metrics import MedVQAEvaluator
from src.rag_system.vector_db import MedicalRetriever
from src.models.qwen_slm import QwenMedVQA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict

class PerformanceMonitor:
    """Monitor system performance during model operations"""
    
    def __init__(self):
        self.peak_vram = 0
        self.peak_ram = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring VRAM and RAM usage"""
        self.monitoring = True
        self.peak_vram = 0
        self.peak_ram = 0
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return peak usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return {
            'peak_vram_gb': self.peak_vram / 1024**3,
            'peak_ram_gb': self.peak_ram / 1024**3
        }
        
    def _monitor_resources(self):
        """Internal monitoring function"""
        while self.monitoring:
            # Monitor VRAM
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated()
                self.peak_vram = max(self.peak_vram, vram_used)
            
            # Monitor RAM
            ram_used = psutil.virtual_memory().used
            self.peak_ram = max(self.peak_ram, ram_used)
            
            time.sleep(0.1)  # Monitor every 100ms

def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }

def calculate_payload_size(model_weights):
    """Calculate the size of model weights in MB"""
    total_size = 0
    for name, param in model_weights.items():
        # Each parameter is a tensor, size = num_elements * 4 bytes (float32)
        param_size = param.numel() * 4
        total_size += param_size
    
    return total_size / (1024 * 1024)  # Convert to MB

def test_fed_llava(num_clients=3, num_rounds=2):
    """Test Fed-LLaVA approach (baseline LLaVA with federated learning)"""
    print("\nTesting Fed-LLaVA")
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Load Train and Test splits separately
    vqa_rad_train = load_from_disk("./data/vqa_rad_full/train")
    path_vqa_train = load_from_disk("./data/path_vqa_full/train")
    vqa_rad_test = load_from_disk("./data/vqa_rad_full/test")
    path_vqa_test = load_from_disk("./data/path_vqa_full/test")
    eval_seed = 42

    # 1. Prepare evaluation sets (using official Test data)
    vqa_rad_shuffled = vqa_rad_test.shuffle(seed=eval_seed)
    path_vqa_shuffled = path_vqa_test.shuffle(seed=eval_seed+1)
    
    eval_size = 100
    vqa_rad_eval = vqa_rad_shuffled.select(range(min(eval_size, len(vqa_rad_test))))
    path_vqa_eval = path_vqa_shuffled.select(range(min(eval_size, len(path_vqa_test))))
    
    print(f"Evaluating with {len(vqa_rad_eval)} VQA-RAD and {len(path_vqa_eval)} PathVQA images (from official TEST set)")
    
    # 2. Combine BOTH datasets for training/splitting
    combined_train = concatenate_datasets([vqa_rad_train, path_vqa_train])
    
    # 3. Use combined dataset for Federated Splitting
    splitter_seed = random.randint(0, 1000000)
    splitter = FederatedDataSplitter(combined_train, num_clients=num_clients, seed=splitter_seed)
    print(f"Training on COMBINED dataset (VQA-RAD + PathVQA) with splitter seed: {splitter_seed}")
    client_datasets = splitter.split_iid()
    
    # Initialize baseline LLaVA-style model (simplified for this test)
    print("Initializing Fed-LLaVA model")
    model = QwenMedVQA(use_4bit=True)  # Using same base but without LoRA for comparison
    
    # Count parameters
    param_stats = count_parameters(model.model)
    
    # Simulate federated rounds
    server = FederatedServer()
    payload_sizes = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}/{num_rounds}")
        
        # Simulate weight collection from clients
        client_weights_list = []
        for i in range(num_clients):
            # Get full model weights (not just LoRA)
            weights = {k: v.clone().cpu() for k, v in model.model.named_parameters()}
            client_weights_list.append(weights)
            
            # Calculate payload size for this client
            payload_size = calculate_payload_size(weights)
            payload_sizes.append(payload_size)
        
        # Aggregate weights
        global_weights = server.aggregate_weights(client_weights_list)
        
        # Load aggregated weights
        model_dict = model.model.state_dict()
        model_dict.update(global_weights)
        model.model.load_state_dict(model_dict)
    
    peak_usage = monitor.stop_monitoring()
    
    return {
        'approach': 'Fed-LLaVA',
        'parameters': param_stats,
        'peak_vram_gb': peak_usage['peak_vram_gb'],
        'peak_ram_gb': peak_usage['peak_ram_gb'],
        'avg_payload_mb': sum(payload_sizes) / len(payload_sizes) if payload_sizes else 0,
        'total_payloads_mb': sum(payload_sizes),
        'num_rounds': num_rounds,
        'num_clients': num_clients
    }

def test_fed_slm(num_clients=3, num_rounds=2):
    """Test Fed-SLM approach (with LoRA)"""
    print("\nTesting Fed-SLM")
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Load Train and Test splits separately
    vqa_rad_train = load_from_disk("./data/vqa_rad_full/train")
    path_vqa_train = load_from_disk("./data/path_vqa_full/train")
    
    # Combine datasets for training
    combined_train = concatenate_datasets([vqa_rad_train, path_vqa_train])
    splitter = FederatedDataSplitter(combined_train, num_clients=num_clients)
    client_datasets = splitter.split_iid()
    
    # Initialize model with LoRA
    print("Initializing Fed-SLM model with LoRA")
    model = QwenMedVQA(use_4bit=True)
    model.model = prepare_model_for_kbit_training(model.model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model.model = get_peft_model(model.model, lora_config)
    
    # Count parameters
    param_stats = count_parameters(model.model)
    
    # Get initial LoRA weights
    initial_lora_weights = {}
    for k, v in get_peft_model_state_dict(model.model).items():
        if getattr(v, "device", None) and v.device.type == 'meta':
            initial_lora_weights[k] = torch.zeros(v.shape, dtype=v.dtype, device='cpu')
        else:
            initial_lora_weights[k] = v.clone().cpu()
    
    # Simulate federated rounds
    server = FederatedServer()
    payload_sizes = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}/{num_rounds}")
        
        # Simulate weight collection from clients (only LoRA weights)
        client_weights_list = []
        for i in range(num_clients):
            # Only LoRA weights are transmitted
            weights = {k: v.clone().cpu() for k, v in get_peft_model_state_dict(model.model).items()}
            client_weights_list.append(weights)
            
            # Calculate payload size (only LoRA weights)
            payload_size = calculate_payload_size(weights)
            payload_sizes.append(payload_size)
        
        # Aggregate LoRA weights
        global_weights = server.aggregate_weights(client_weights_list)
        
        # Load aggregated LoRA weights
        model.model.load_state_dict(global_weights, strict=False)
    
    peak_usage = monitor.stop_monitoring()
    
    return {
        'approach': 'Fed-SLM',
        'parameters': param_stats,
        'peak_vram_gb': peak_usage['peak_vram_gb'],
        'peak_ram_gb': peak_usage['peak_ram_gb'],
        'avg_payload_mb': sum(payload_sizes) / len(payload_sizes) if payload_sizes else 0,
        'total_payloads_mb': sum(payload_sizes),
        'num_rounds': num_rounds,
        'num_clients': num_clients
    }

def test_fed_slm_rag(num_clients=3, num_rounds=2):
    """Test Fed-SLM + RAG approach (current full model)"""
    print("\nTesting Fed-SLM + RAG")
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Load Train and Test splits separately
    vqa_rad_train = load_from_disk("./data/vqa_rad_full/train")
    path_vqa_train = load_from_disk("./data/path_vqa_full/train")
    
    # Combine datasets for training
    combined_train = concatenate_datasets([vqa_rad_train, path_vqa_train])
    splitter = FederatedDataSplitter(combined_train, num_clients=num_clients)
    client_datasets = splitter.split_iid()
    
    # Initialize model with LoRA
    print("Initializing Fed-SLM + RAG model...")
    model = QwenMedVQA(use_4bit=True)
    model.model = prepare_model_for_kbit_training(model.model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model.model = get_peft_model(model.model, lora_config)
    
    # Build RAG databases (additional memory overhead)
    print("Building RAG vector databases...")
    eval_seed = 42
    # Build RAG pool from Training data only
    vqa_rad_rag_pool = vqa_rad_train.shuffle(seed=eval_seed+2).select(range(min(2000, len(vqa_rad_train))))
    path_vqa_rag_pool = path_vqa_train.shuffle(seed=eval_seed+3).select(range(min(2000, len(path_vqa_train))))
    
    retriever_vr = MedicalRetriever(); retriever_vr.build_index_from_dataset(vqa_rad_rag_pool)
    retriever_pv = MedicalRetriever(); retriever_pv.build_index_from_dataset(path_vqa_rag_pool)
    
    # Count parameters
    param_stats = count_parameters(model.model)
    
    # Get initial LoRA weights
    initial_lora_weights = {}
    for k, v in get_peft_model_state_dict(model.model).items():
        if getattr(v, "device", None) and v.device.type == 'meta':
            initial_lora_weights[k] = torch.zeros(v.shape, dtype=v.dtype, device='cpu')
        else:
            initial_lora_weights[k] = v.clone().cpu()
    
    # Simulate federated rounds
    server = FederatedServer()
    payload_sizes = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}/{num_rounds}")
        
        # Simulate weight collection from clients (only LoRA weights)
        client_weights_list = []
        for i in range(num_clients):
            # Only LoRA weights are transmitted
            weights = {k: v.clone().cpu() for k, v in get_peft_model_state_dict(model.model).items()}
            client_weights_list.append(weights)
            
            # Calculate payload size (only LoRA weights)
            payload_size = calculate_payload_size(weights)
            payload_sizes.append(payload_size)
        
        # Aggregate LoRA weights
        global_weights = server.aggregate_weights(client_weights_list)
        
        # Load aggregated LoRA weights
        model.model.load_state_dict(global_weights, strict=False)
    
    peak_usage = monitor.stop_monitoring()
    
    return {
        'approach': 'Fed-SLM + RAG',
        'parameters': param_stats,
        'peak_vram_gb': peak_usage['peak_vram_gb'],
        'peak_ram_gb': peak_usage['peak_ram_gb'],
        'avg_payload_mb': sum(payload_sizes) / len(payload_sizes) if payload_sizes else 0,
        'total_payloads_mb': sum(payload_sizes),
        'num_rounds': num_rounds,
        'num_clients': num_clients
    }

def save_comparison_results(results, filename="computational_comparison.json"):
    """Save comparison results to JSON file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    comparison_data = {
        "timestamp": timestamp,
        "experiment_config": {
            "num_clients": results[0]['num_clients'] if results else 0,
            "num_rounds": results[0]['num_rounds'] if results else 0
        },
        "results": results,
        "summary": {
            "parameter_efficiency": {},
            "memory_efficiency": {},
            "communication_efficiency": {}
        }
    }
    
    # Calculate summary statistics
    if len(results) >= 2:
        # Parameter efficiency (trainable parameters / total parameters)
        for result in results:
            approach = result['approach']
            params = result['parameters']
            if params['total_parameters'] > 0:
                efficiency = (params['trainable_parameters'] / params['total_parameters']) * 100
                comparison_data["summary"]["parameter_efficiency"][approach] = round(efficiency, 2)
            else:
                comparison_data["summary"]["parameter_efficiency"][approach] = 0.0
        
        # Memory efficiency (inverse of VRAM usage)
        vram_values = [r['peak_vram_gb'] for r in results if r['peak_vram_gb'] > 0]
        if vram_values:  # Check if list is not empty
            min_vram = min(vram_values)
            for result in results:
                approach = result['approach']
                if result['peak_vram_gb'] > 0:
                    efficiency = min_vram / result['peak_vram_gb'] * 100
                    comparison_data["summary"]["memory_efficiency"][approach] = round(efficiency, 2)
                else:
                    comparison_data["summary"]["memory_efficiency"][approach] = 0.0
        else:
            # All VRAM values are zero, set all to 0
            for result in results:
                comparison_data["summary"]["memory_efficiency"][result['approach']] = 0.0
        
        # Communication efficiency (inverse of payload size)
        payload_values = [r['avg_payload_mb'] for r in results if r['avg_payload_mb'] > 0]
        if payload_values:  # Check if list is not empty
            min_payload = min(payload_values)
            for result in results:
                approach = result['approach']
                if result['avg_payload_mb'] > 0:
                    efficiency = min_payload / result['avg_payload_mb'] * 100
                    comparison_data["summary"]["communication_efficiency"][approach] = round(efficiency, 2)
                else:
                    comparison_data["summary"]["communication_efficiency"][approach] = 0.0
        else:
            # All payload values are zero, set all to 0
            for result in results:
                comparison_data["summary"]["communication_efficiency"][result['approach']] = 0.0
    
    # Save to file
    os.makedirs("./data", exist_ok=True)
    filepath = os.path.join("./data", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {filepath}")
    return filepath

def print_comparison_table(results):
    """Print a formatted comparison table"""
    print("\n" + "="*100)
    print("COMPUTATIONAL COMPARISON TABLE")
    print("="*100)
    
    header = f"{'Approach':<20} {'Total Params':<15} {'Trainable':<12} {'VRAM (GB)':<12} {'RAM (GB)':<12} {'Payload (MB)':<15}"
    print(header)
    print("-" * 100)
    
    for result in results:
        approach = result['approach']
        total_params = f"{result['parameters']['total_parameters']:,}"
        trainable_params = f"{result['parameters']['trainable_parameters']:,}"
        vram = f"{result['peak_vram_gb']:.2f}"
        ram = f"{result['peak_ram_gb']:.2f}"
        payload = f"{result['avg_payload_mb']:.2f}"
        
        row = f"{approach:<20} {total_params:<15} {trainable_params:<12} {vram:<12} {ram:<12} {payload:<15}"
        print(row)
    
    print("="*100)

def get_user_setup():
    """Get user configuration for comparison"""
    print("\n=== Computational Comparison Setup ===")
    
    while True:
        try:
            num_clients = int(input("1. Enter number of clients (e.g., 3, 5): "))
            if num_clients >= 2: break
            else: print("At least 2 clients required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        try:
            num_rounds = int(input("2. Enter number of rounds (e.g., 2, 3): "))
            if num_rounds >= 1: break
            else: print("At least 1 round required!")
        except ValueError: print("Please enter a valid integer!")
    
    return num_clients, num_rounds

if __name__ == "__main__":
    print("=== Federated Learning Computational Comparison ===")
    print("Comparing: Fed-LLaVA vs Fed-SLM vs Fed-SLM+RAG")
    
    # Get user configuration
    clients, rounds = get_user_setup()
    
    print(f"\nStarting comparison with {clients} clients and {rounds} rounds...")
    print("This may take several minutes...\n")
    
    results = []
    
    try:
        # Test each approach
        result1 = test_fed_llava(clients, rounds)
        results.append(result1)
        
        result2 = test_fed_slm(clients, rounds)
        results.append(result2)
        
        result3 = test_fed_slm_rag(clients, rounds)
        results.append(result3)
        
        # Print results
        print_comparison_table(results)
        
        # Save results
        save_comparison_results(results)
        
        print("\n=== Comparison Complete! ===")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        print("Please check your setup and try again.")