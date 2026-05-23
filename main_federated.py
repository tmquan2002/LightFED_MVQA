import json
import os
# Fix duplicate OpenMP runtime library error on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import time
import torch
import random
from datasets import load_dataset
from src.data_processing.data_splitter import FederatedDataSplitter
from src.federated.server import FederatedServer
from src.evaluation.metrics import MedVQAEvaluator
from src.rag_system.vector_db import MedicalRetriever
from src.models.qwen_slm import QwenMedVQA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict

class VirtualClient:
    """Virtual Hospital: Holds Data and personal LoRA weights"""
    def __init__(self, client_id, local_dataset, initial_weights):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.lora_weights = {k: v.clone() for k, v in initial_weights.items()}

    def train_local(self, shared_slm, epochs=1):
        import torch.optim as optim
        from qwen_vl_utils import process_vision_info
        
        print(f"  [{self.client_id}] Training locally for {epochs} epoch(s)...")
        model = shared_slm.model
        processor = shared_slm.processor
        device = shared_slm.device
        
        model.load_state_dict(self.lora_weights, strict=False)
        model.train()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
        optimizer.zero_grad()
        
        total_loss = 0.0
        steps = 0
        
        indices = list(range(len(self.local_dataset)))
        
        for epoch in range(epochs):
            import random
            random.shuffle(indices)
            for i, idx in enumerate(indices):
                sample = self.local_dataset[idx]
                question = sample['question']
                answer = str(sample['answer']).lower()
                
                img_obj = shared_slm._preprocess_image(sample['image'])
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_obj},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # Copy input_ids to labels for Causal LM loss
                inputs['labels'] = inputs['input_ids'].clone()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs.loss / 4  # Gradient accumulation scale (steps=4)
                
                loss.backward()
                
                if (i + 1) % 4 == 0 or (i + 1) == len(indices):
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * 4  # Restore original loss scale for stats
                steps += 1
                
                if i % 10 == 0:
                    print(f"    [{self.client_id}] Epoch {epoch+1}/{epochs} | Step {i}/{len(indices)} | Loss: {loss.item():.4f}   ", end="\r")
                    
        print()  # newline after progress
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        
        # Save updated LoRA weights
        raw_weights = get_peft_model_state_dict(model)
        self.lora_weights = {}
        for name, param in raw_weights.items():
            if getattr(param, "device", None) and param.device.type == 'meta':
                self.lora_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
            else:
                self.lora_weights[name] = param.clone().detach().cpu()
                
        return self.lora_weights, avg_loss

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_scores_for_json(c_scores, o_scores):
    return {
        "Accuracy": round(c_scores.get('Accuracy', 0) * 100, 1),
        "F1-Score": round(c_scores.get('F1-Score', 0) * 100, 1),
        "BLEU": round(o_scores.get('BLEU', 0) * 100, 1),
        "ROUGE-L": round(o_scores.get('ROUGE-L', 0) * 100, 1)
    }

def evaluate_dataset(shared_slm, dataset, evaluator, retriever=None):
    shared_slm.model.eval() 
    closed_preds, closed_refs, open_preds, open_refs = [], [], [], []
    test_samples = dataset
    total = len(test_samples)
    
    start_infer_time = time.time() # START INFERENCE TIMER
    
    for i, sample in enumerate(test_samples):
        print(f"Evaluating: {i+1}/{total} images...", end="\r")
        question = sample['question']
        ground_truth = str(sample['answer']).lower()
        image = sample['image']
        
        if retriever:
            # Use a more focused c=5 to avoid distracting the SLM
            similar_cases = retriever.search_similar_cases(image, c=5)
            
            context_text = "### Medical Reference Cases:\n" if similar_cases else ""
            for j, case in enumerate(similar_cases):
                context_text += f"Case {j+1}: Q: '{case['question']}' -> A: '{case['answer']}'\n"
            
            # Optimized Prompt for Medical SLM
            augmented_question = (
                f"{context_text}\n"
                "### Instruction:\n"
                "You are a medical expert. Based on the provided image and the similar reference cases above, "
                f"answer the following question concisely.\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
            pred = shared_slm.predict(image, augmented_question)
        else:
            # Concise prompt for direct prediction
            direct_prompt = f"Answer this medical question concisely based on the image: {question}\nAnswer:"
            pred = shared_slm.predict(image, direct_prompt)
            
        if ground_truth in ['yes', 'no'] or len(ground_truth.split()) <= 2:
            closed_preds.append(pred)
            closed_refs.append(ground_truth)
        else:
            open_preds.append(pred)
            open_refs.append(ground_truth)
            
    end_infer_time = time.time() # END INFERENCE TIMER
    infer_time = round(end_infer_time - start_infer_time, 2)
    print(f"\nInference Time: {infer_time} seconds")
    
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs), infer_time

def run_federated_simulation(num_clients, num_rounds, epochs, split_type, alpha):
    print(f"\nSTARTING SEPARATE FEDERATED EXPERIMENTS: {num_clients} Clients | {num_rounds} Rounds | {epochs} Epochs | {split_type.upper()} | Alpha = {alpha if split_type == 'non-iid' else 'NA'}")
    
    # Load datasets
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    
    vqa_rad_train = vqa_rad["train"]
    path_vqa_train = path_vqa["train"]
    vqa_rad_test = vqa_rad["test"]
    path_vqa_test = path_vqa["test"]
    
    # Use time-based random sampling for evaluation
    random.seed(int(time.time()))
    eval_seed = random.randint(0, 1000000)
    
    print(f"Using random evaluation seed: {eval_seed}")
    
    # 1. Prepare evaluation sets (from official Test splits)
    vqa_rad_shuffled = vqa_rad_test.shuffle(seed=eval_seed)
    path_vqa_shuffled = path_vqa_test.shuffle(seed=eval_seed+1)
    
    eval_size = 100
    vqa_rad_eval = vqa_rad_shuffled.select(range(min(eval_size, len(vqa_rad_test))))
    path_vqa_eval = path_vqa_shuffled.select(range(min(eval_size, len(path_vqa_test))))
    
    print(f"Evaluating with {len(vqa_rad_eval)} VQA-RAD and {len(path_vqa_eval)} PathVQA images (from official TEST set)")
    
    evaluator = MedVQAEvaluator()
    server = FederatedServer()

    # 2. Build RAG Vector Databases (FAISS) - LEAKAGE FREE
    print("\n[RAG] Building Vector Databases from Training data only...")
    vqa_rad_rag_pool = vqa_rad_train
    path_vqa_rag_pool = path_vqa_train.shuffle(seed=eval_seed+2).select(range(min(5000, len(path_vqa_train))))
    
    retriever_vr = MedicalRetriever(); retriever_vr.build_index_from_dataset(vqa_rad_rag_pool)
    retriever_pv = MedicalRetriever(); retriever_pv.build_index_from_dataset(path_vqa_rag_pool)
    
    os.makedirs("./data", exist_ok=True)
    alpha_str = str(alpha) if split_type == 'non-iid' else "NA"
    file_name = f"eval_results_{num_clients}clients_{num_rounds}rounds_{split_type.upper()}_a{alpha_str}.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Num_Clients": num_clients,
            "Max_Rounds_Configured": num_rounds,
            "Local_Epochs": epochs,
            "Split_Type": split_type.upper(),
            "Alpha": alpha if split_type == 'non-iid' else "NA"
        },
        "Training_Stats": {},
        "Results": {
            "Fed-SLM (No RAG)": {},
            "Proposed (Fed+RAG)": {}
        }
    }

    def save_current_progress(phase_name):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved progress '{phase_name}' to: {file_name}")

    print("\n>>> INITIALIZING SHARED QWEN2-VL ENGINE...")
    shared_slm = QwenMedVQA(use_4bit=True)
    shared_slm.model = prepare_model_for_kbit_training(shared_slm.model)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    shared_slm.model = get_peft_model(shared_slm.model, lora_config)
    
    initial_lora_weights = {}
    for k, v in get_peft_model_state_dict(shared_slm.model).items():
        if getattr(v, "device", None) and v.device.type == 'meta':
            initial_lora_weights[k] = torch.zeros(v.shape, dtype=v.dtype, device='cpu')
        else:
            initial_lora_weights[k] = v.clone().cpu()

    # ==================== EXPERIMENT 1: VQA-RAD (Radiology) ====================
    print(f"\n==================== EXPERIMENT 1: VQA-RAD ====================")
    splitter_seed = random.randint(0, 1000000)
    splitter_vr = FederatedDataSplitter(vqa_rad_train, num_clients=num_clients, seed=splitter_seed)
    
    if split_type == 'iid':
        client_datasets_vr = splitter_vr.split_iid()
    else:
        client_datasets_vr = splitter_vr.split_non_iid(alpha=alpha)
        
    total_samples_vr = sum(len(ds) for ds in client_datasets_vr)
    contributions_vr = {f"Hospital_{i+1}": round((len(ds) / total_samples_vr) * 100, 2) for i, ds in enumerate(client_datasets_vr)}
    
    clients = []
    for i in range(num_clients):
        clients.append(VirtualClient(f"Hospital_{i+1}", client_datasets_vr[i], initial_lora_weights))
        
    global_weights = None
    best_loss = float('inf')
    patience = 3  
    patience_counter = 0
    actual_rounds = 0
    final_loss = 0.0

    start_train_time = time.time()
    for round_num in range(1, num_rounds + 1):
        print(f"  [VQA-RAD] Round {round_num}/{num_rounds}: Training...")
        client_weights_list = []
        client_losses = []
        
        for client in clients:
            weights, loss = client.train_local(shared_slm, epochs=epochs)
            client_weights_list.append(weights)
            client_losses.append(loss)
            
        client_sizes = [len(c.local_dataset) for c in clients]
        global_weights = server.aggregate_weights(client_weights_list, client_sizes=client_sizes)
        for client in clients:
            client.lora_weights = {k: v.clone() for k, v in global_weights.items()}
            
        avg_loss = sum(client_losses) / len(client_losses)
        final_loss = avg_loss
        actual_rounds = round_num
        print(f"  [VQA-RAD] Round {round_num} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [VQA-RAD] Early stopping triggered.")
                break
                
    total_train_time_vr = round(time.time() - start_train_time, 2)
    results_dict["Training_Stats"]["VQA-RAD"] = {
        "Client_Data_Contributions_Percent": contributions_vr,
        "Actual_Rounds_Run": actual_rounds,
        "Final_Average_Loss": round(final_loss, 4),
        "Training_Time_Seconds": total_train_time_vr
    }
    
    # Evaluate VQA-RAD
    shared_slm.model.load_state_dict(global_weights, strict=False)
    print("\nEvaluating VQA-RAD Scenarios...")
    vr_c_norag, vr_o_norag, vr_t_norag = evaluate_dataset(shared_slm, vqa_rad_eval, evaluator, None)
    vr_c_rag, vr_o_rag, vr_t_rag = evaluate_dataset(shared_slm, vqa_rad_eval, evaluator, retriever_vr)
    
    results_dict["Results"]["Fed-SLM (No RAG)"]["VQA-RAD"] = format_scores_for_json(vr_c_norag, vr_o_norag)
    results_dict["Results"]["Proposed (Fed+RAG)"]["VQA-RAD"] = format_scores_for_json(vr_c_rag, vr_o_rag)
    save_current_progress("VQA-RAD Experiment Completed")

    # ==================== EXPERIMENT 2: PathVQA (Pathology) ====================
    print(f"\n==================== EXPERIMENT 2: PathVQA ====================")
    # Reset model parameters to base initial state
    shared_slm.model.load_state_dict(initial_lora_weights, strict=False)
    
    splitter_seed_pv = random.randint(0, 1000000)
    splitter_pv = FederatedDataSplitter(path_vqa_train, num_clients=num_clients, seed=splitter_seed_pv)
    
    if split_type == 'iid':
        client_datasets_pv = splitter_pv.split_iid()
    else:
        client_datasets_pv = splitter_pv.split_non_iid(alpha=alpha)
        
    total_samples_pv = sum(len(ds) for ds in client_datasets_pv)
    contributions_pv = {f"Hospital_{i+1}": round((len(ds) / total_samples_pv) * 100, 2) for i, ds in enumerate(client_datasets_pv)}
    
    clients = []
    for i in range(num_clients):
        clients.append(VirtualClient(f"Hospital_{i+1}", client_datasets_pv[i], initial_lora_weights))
        
    global_weights = None
    best_loss = float('inf')
    patience = 3  
    patience_counter = 0
    actual_rounds = 0
    final_loss = 0.0

    start_train_time = time.time()
    for round_num in range(1, num_rounds + 1):
        print(f"  [PathVQA] Round {round_num}/{num_rounds}: Training...")
        client_weights_list = []
        client_losses = []
        
        for client in clients:
            weights, loss = client.train_local(shared_slm, epochs=epochs)
            client_weights_list.append(weights)
            client_losses.append(loss)
            
        client_sizes = [len(c.local_dataset) for c in clients]
        global_weights = server.aggregate_weights(client_weights_list, client_sizes=client_sizes)
        for client in clients:
            client.lora_weights = {k: v.clone() for k, v in global_weights.items()}
            
        avg_loss = sum(client_losses) / len(client_losses)
        final_loss = avg_loss
        actual_rounds = round_num
        print(f"  [PathVQA] Round {round_num} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [PathVQA] Early stopping triggered.")
                break
                
    total_train_time_pv = round(time.time() - start_train_time, 2)
    results_dict["Training_Stats"]["PathVQA"] = {
        "Client_Data_Contributions_Percent": contributions_pv,
        "Actual_Rounds_Run": actual_rounds,
        "Final_Average_Loss": round(final_loss, 4),
        "Training_Time_Seconds": total_train_time_pv
    }
    
    # Evaluate PathVQA
    shared_slm.model.load_state_dict(global_weights, strict=False)
    print("\nEvaluating PathVQA Scenarios...")
    pv_c_norag, pv_o_norag, pv_t_norag = evaluate_dataset(shared_slm, path_vqa_eval, evaluator, None)
    pv_c_rag, pv_o_rag, pv_t_rag = evaluate_dataset(shared_slm, path_vqa_eval, evaluator, retriever_pv)
    
    results_dict["Results"]["Fed-SLM (No RAG)"]["PathVQA"] = format_scores_for_json(pv_c_norag, pv_o_norag)
    results_dict["Results"]["Fed-SLM (No RAG)"]["Inference_Time_Seconds"] = round(vr_t_norag + pv_t_norag, 2)
    
    results_dict["Results"]["Proposed (Fed+RAG)"]["PathVQA"] = format_scores_for_json(pv_c_rag, pv_o_rag)
    results_dict["Results"]["Proposed (Fed+RAG)"]["Inference_Time_Seconds"] = round(vr_t_rag + pv_t_rag, 2)
    
    save_current_progress("All Experiments Completed")
    print(f"\nCOMPLETED! Evaluation metrics securely saved to: {json_path}")

    print(f"\nCOMPLETED! Evaluation metrics securely saved to: {json_path}")

def get_user_setup():
    
    while True:
        try:
            num_clients = int(input("\n1. Enter the number of participating Hospitals (e.g., 2, 3, 5): "))
            if num_clients >= 2: break
            else: print("At least 2 Hospitals are required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        try:
            num_rounds = int(input("2. Enter the max communication rounds (e.g., 5, 10, 20): "))
            if num_rounds >= 1: break
            else: print("At least 1 round is required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        try:
            epochs = int(input("3. Enter local Epochs for each Hospital (e.g., 1, 3, 5): "))
            if epochs >= 1: break
            else: print("At least 1 epoch is required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        split_type = input("4. Select data splitting mode ('iid' or 'non-iid'): ").strip().lower()
        if split_type in ['iid', 'non-iid']: break
        else: print("Only 'iid' or 'non-iid' are accepted!")

    alpha = 0.5 
    if split_type == 'non-iid':
        while True:
            try:
                alpha = float(input("5. Enter Alpha coefficient (e.g., 0.1 for extreme non-IID, 0.5 for moderate): "))
                if alpha > 0: break
                else: print("lpha must be greater than 0!")
            except ValueError: print("Please enter a valid float number!")

    return num_clients, num_rounds, epochs, split_type, alpha

if __name__ == "__main__":
    clients_input, rounds_input, epochs_input, split_input, alpha_input = get_user_setup()
    run_federated_simulation(clients_input, rounds_input, epochs_input, split_input, alpha_input)