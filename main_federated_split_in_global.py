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
    """Virtual Hospital: Holds Data, personal LoRA weights, and local RAG retriever"""
    def __init__(self, client_id, local_dataset, initial_weights, dataset_name):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.lora_weights = {k: v.clone() for k, v in initial_weights.items()}
        
        # Initialize local retriever for leak-free RAG training
        print(f"  [{self.client_id}] Initializing local retriever indexing {len(local_dataset)} samples...")
        self.retriever = MedicalRetriever(f"{dataset_name}_{client_id}")
        self.retriever.train_dataset = local_dataset
        self.retriever.build_index_from_dataset(local_dataset)

    def train_local(self, shared_slm, epochs=1, max_steps=100):
        import torch.optim as optim
        from qwen_vl_utils import process_vision_info
        
        print(f"  [{self.client_id}] Training locally for {epochs} epoch(s)...")
        model = shared_slm.model
        processor = shared_slm.processor
        device = shared_slm.device
        
        model.load_state_dict(self.lora_weights, strict=False)
        model.train()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        optimizer.zero_grad()
        
        total_loss = 0.0
        steps = 0
        
        indices = list(range(len(self.local_dataset)))
        
        # Helper to resize/optimize images to avoid VRAM/RAM overload
        def optimize_img(pil_img, max_res=336):
            from PIL import Image
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)
            return pil_img

        for epoch in range(epochs):
            random.shuffle(indices)
            epoch_indices = indices[:max_steps]  # Cap training samples per round
            print(f"    [{self.client_id}] Using {len(epoch_indices)}/{len(indices)} samples this round")
            for i, idx in enumerate(epoch_indices):
                sample = self.local_dataset[idx]
                question = sample['question']
                answer = str(sample['answer']).lower()
                
                # Dynamic retrieval: query local retriever for Top-4 cases (excluding query itself)
                retrieved = self.retriever.search_similar_cases(sample['image'], query_question=question, c=4)
                similar_cases = [case for case in retrieved if case['id'] != idx][:3]
                
                # Format prompts exactly using the multi-turn RAG template
                messages = []
                for j, case in enumerate(similar_cases):
                    try:
                        ref_img = shared_slm._preprocess_image(case['image'])
                        ref_img = optimize_img(ref_img)
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": ref_img},
                                {"type": "text", "text": f"Reference Case {j+1}:\nQuestion: {case['question']}"}
                            ]
                        })
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"Answer: {case['answer']}"}
                            ]
                        })
                    except Exception:
                        continue
                
                target_img = shared_slm._preprocess_image(sample['image'])
                target_img = optimize_img(target_img)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": target_img},
                        {"type": "text", "text": f"Current Question: {question}\nBased on the reference cases above, what is the correct answer for the current case? Answer concisely."}
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                })
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # Copy input_ids to labels and mask out prompt tokens (set to -100)
                prompt_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_inputs = processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                prompt_len = prompt_inputs['input_ids'].shape[1]
                
                labels = inputs['input_ids'].clone()
                labels[:, :prompt_len] = -100
                inputs['labels'] = labels
                
                # Mixed precision forward pass with GradScaler
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = model(**inputs)
                        loss = outputs.loss / 4  # Gradient accumulation scale (steps=4)
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % 4 == 0 or (i + 1) == len(epoch_indices):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss / 4
                    loss.backward()
                    
                    if (i + 1) % 4 == 0 or (i + 1) == len(epoch_indices):
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * 4  # Restore original loss scale for stats
                steps += 1
                
                if i % 10 == 0:
                    print(f"    [{self.client_id}] Epoch {epoch+1}/{epochs} | Step {i}/{len(epoch_indices)} | Loss: {loss.item():.4f}   ", end="\r")
                    
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

def format_scores_for_json(c_scores, o_scores, question_type="all"):
    # If training on open-ended questions, we evaluate Accuracy (closed-ended test set)
    # If training on closed-ended (Yes-No) questions, we evaluate GA (open-ended test set)
    accuracy = round(c_scores.get('Accuracy', 0) * 100, 1) if question_type in ["all", "open"] else 0.0
    f1 = round(c_scores.get('F1-Score', 0) * 100, 1) if question_type in ["all", "open"] else 0.0
    bleu = round(o_scores.get('BLEU', 0) * 100, 1) if question_type in ["all", "closed"] else 0.0
    rouge = round(o_scores.get('ROUGE-L', 0) * 100, 1) if question_type in ["all", "closed"] else 0.0
    return {
        "Accuracy": accuracy,
        "F1-Score": f1,
        "BLEU": bleu,
        "ROUGE-L": rouge
    }

def evaluate_dataset(shared_slm, dataset, evaluator, retriever, question_type="all"):
    shared_slm.model.eval()
    closed_preds, closed_refs, open_preds, open_refs = [], [], [], []
    total = len(dataset)
    start_infer_time = time.time()
    
    for i, sample in enumerate(dataset):
        print(f"    Evaluating: {i+1}/{total} images...", end="\r")
        question = sample['question']
        ground_truth = str(sample['answer']).lower()
        image = sample['image']
        
        # Option A: Use the unified helper function from FederatedDataSplitter
        is_closed = FederatedDataSplitter.is_closed_ended(ground_truth)
        
        # For accuracy, only train on open ended questions, while GA train on Yes-No questions:
        # - If training on open-ended questions (question_type == "open"), we evaluate accuracy on closed-ended (Yes-No) questions, so skip open-ended evaluation queries.
        # - If training on closed-ended questions (question_type == "closed"), we evaluate GA on open-ended questions, so skip closed-ended evaluation queries.
        if question_type == "open" and not is_closed:
            continue
        elif question_type == "closed" and is_closed:
            continue
            
        # RAG prediction
        similar_cases = retriever.search_similar_cases(image, query_question=question, c=3)
        pred = shared_slm.predict(image, question, retrieved_cases=similar_cases)
        
        if is_closed:
            closed_preds.append(pred); closed_refs.append(ground_truth)
        else:
            open_preds.append(pred); open_refs.append(ground_truth)
            
    infer_time = round(time.time() - start_infer_time, 2)
    print(f"\nInference Time: {infer_time} seconds")
    
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs), infer_time

def run_federated_simulation(num_clients, num_rounds, epochs, split_type, alpha, question_type="all", max_samples=None):
    print(f"\nSTARTING SEPARATE FEDERATED EXPERIMENTS (WITH RAG): {num_clients} Clients | {num_rounds} Rounds | {epochs} Epochs | {split_type.upper()} | Alpha = {alpha if split_type == 'non-iid' else 'NA'} | Question Type = {question_type.upper()} | Max Samples = {max_samples if max_samples is not None else 'ALL'}")
    
    # Temporarily don't experiment on VQA-RAD
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    
    path_vqa_train = path_vqa["train"]
    path_vqa_test = path_vqa["test"]
    
    # Use time-based random sampling for evaluation
    random.seed(int(time.time()))
    eval_seed = random.randint(0, 1000000)
    print(f"Using random evaluation seed: {eval_seed}")
    
    path_vqa_shuffled = path_vqa_test.shuffle(seed=eval_seed+1)
    
    eval_size = 50
    path_vqa_eval = path_vqa_shuffled.select(range(min(eval_size, len(path_vqa_test))))
    
    print(f"Evaluating with {len(path_vqa_eval)} PathVQA images (from official TEST set)")
    
    evaluator = MedVQAEvaluator()
    server = FederatedServer()

    # Build/Load RAG Vector Databases (FAISS) for global evaluation
    print("\n[RAG] Loading or Building Vector Databases from Training data only...")
    
    retriever_pv = MedicalRetriever("pathvqa")
    retriever_pv.train_dataset = path_vqa_train
    retriever_pv.build_index_from_dataset(path_vqa_train)
    
    os.makedirs("./data", exist_ok=True)
    alpha_str = str(alpha) if split_type == 'non-iid' else "NA"
    file_name = f"eval_results_{num_clients}clients_{num_rounds}rounds_{split_type.upper()}_a{alpha_str}_optA.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Num_Clients": num_clients,
            "Max_Rounds_Configured": num_rounds,
            "Local_Epochs": epochs,
            "Split_Type": split_type.upper(),
            "Alpha": alpha if split_type == 'non-iid' else "NA",
            "Model_Type": "Federated Learning with RAG (Unified Check)"
        },
        "Training_Stats": {},
        "Results": {
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

    # ==================== EXPERIMENT 1: VQA-RAD ====================
    print(f"\n==================== EXPERIMENT 1: VQA-RAD (TEMPORARILY SKIPPED) ====================")
    vr_t = 0.0
    results_dict["Training_Stats"]["VQA-RAD"] = "Skipped"
    results_dict["Results"]["Proposed (Fed+RAG)"]["VQA-RAD"] = {
        "Accuracy": 0.0,
        "F1-Score": 0.0,
        "BLEU": 0.0,
        "ROUGE-L": 0.0
    }
    save_current_progress("VQA-RAD Experiment Skipped")

    # ==================== EXPERIMENT 2: PathVQA ====================
    print(f"\n==================== EXPERIMENT 2: PathVQA ====================")
    # Reset model parameters to base initial state
    shared_slm.model.load_state_dict(initial_lora_weights, strict=False)
    
    splitter_seed_pv = random.randint(0, 1000000)
    splitter_pv = FederatedDataSplitter(path_vqa_train, num_clients=num_clients, seed=splitter_seed_pv, question_type=question_type, max_samples=max_samples)
    
    if split_type == 'iid':
        client_datasets_pv = splitter_pv.split_iid()
    else:
        client_datasets_pv = splitter_pv.split_non_iid(alpha=alpha)
        
    total_samples_pv = sum(len(ds) for ds in client_datasets_pv)
    contributions_pv = {f"Hospital_{i+1}": round((len(ds) / total_samples_pv) * 100, 2) for i, ds in enumerate(client_datasets_pv)}
    
    checkpoint_pv = f"./model_checkpoints/lora_pv_{num_clients}clients_{num_rounds}rounds_{epochs}epochs_{split_type}_a{alpha_str}_optA.pt"

    global_weights = None
    start_round = 1
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    actual_rounds = 0
    final_loss = 0.0
    total_train_time_pv = 0.0
    start_train_time = time.time()

    if os.path.exists(checkpoint_pv):
        print(f"\n[LOAD] Found existing trained weights checkpoint for PathVQA: {checkpoint_pv}")
        checkpoint = torch.load(checkpoint_pv, map_location='cpu')
        if isinstance(checkpoint, dict) and 'weights' in checkpoint:
            global_weights = checkpoint['weights']
            start_round = checkpoint.get('round', num_rounds) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
        else:
            global_weights = checkpoint
            start_round = num_rounds + 1
            
        if start_round > num_rounds:
            print("Loading saved weights and skipping training...")
            actual_rounds = "Loaded from checkpoint"
            final_loss = best_loss if best_loss != float('inf') else 0.0
        else:
            print(f"Resuming training from Round {start_round}...")

    clients = []
    if start_round <= num_rounds:
        for i in range(num_clients):
            initial_client_weights = global_weights if global_weights is not None else initial_lora_weights
            clients.append(VirtualClient(f"Hospital_{i+1}", client_datasets_pv[i], initial_client_weights, "pathvqa"))
            
        for round_num in range(start_round, num_rounds + 1):
            print(f"  [PathVQA] Round {round_num}/{num_rounds}: Training...")
            client_weights_list = []
            client_losses = []
            
            for client in clients:
                weights, loss = client.train_local(shared_slm, epochs=epochs, max_steps=100)
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
                
            # AUTO-SAVE checkpoint every round
            torch.save({
                'round': round_num,
                'weights': global_weights,
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }, checkpoint_pv)
            print(f"  [PathVQA] Auto-saved Round {round_num} checkpoint.")

            if patience_counter >= patience:
                print(f"  [PathVQA] Early stopping triggered.")
                break
                
        total_train_time_pv = round(time.time() - start_train_time, 2)
        
    results_dict["Training_Stats"]["PathVQA"] = {
        "Client_Data_Contributions_Percent": contributions_pv,
        "Actual_Rounds_Run": actual_rounds,
        "Final_Average_Loss": round(final_loss, 4) if isinstance(final_loss, (int, float)) else final_loss,
        "Training_Time_Seconds": total_train_time_pv
    }
    
    # Evaluate PathVQA
    shared_slm.model.load_state_dict(global_weights, strict=False)
    print("\nEvaluating PathVQA (Proposed Fed+RAG)...")
    pv_c, pv_o, pv_t = evaluate_dataset(shared_slm, path_vqa_eval, evaluator, retriever_pv, question_type=question_type)
    
    results_dict["Results"]["Proposed (Fed+RAG)"]["PathVQA"] = format_scores_for_json(pv_c, pv_o, question_type=question_type)
    results_dict["Results"]["Proposed (Fed+RAG)"]["Inference_Time_Seconds"] = round(vr_t + pv_t, 2)
    save_current_progress("All Experiments Completed")
    print(f"\nCOMPLETED! Evaluation metrics saved to: {json_path}")

    if 'clients' in locals():
        del clients
    del client_datasets_pv
    clear_memory()

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
            epochs = int(input("3. Enter local Epochs for each Hospital (e.g., 1, 2, 3 - 1 is recommended for speed/stability): "))
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
                else: print("Alpha must be greater than 0!")
            except ValueError: print("Please enter a valid float number!")

    while True:
        question_type = input("6. Select question type to train on ('all', 'closed', 'open'): ").strip().lower()
        if question_type in ['all', 'closed', 'open']: break
        else: print("Only 'all', 'closed', or 'open' are accepted!")

    max_samples = None
    while True:
        max_samples_input = input("7. Enter max training samples per dataset (e.g., 1000, or press Enter for all): ").strip()
        if max_samples_input == "":
            max_samples = None
            break
        try:
            max_samples = int(max_samples_input)
            if max_samples >= 1: break
            else: print("Must be at least 1!")
        except ValueError:
            print("Please enter a valid integer or press Enter!")

    return num_clients, num_rounds, epochs, split_type, alpha, question_type, max_samples

if __name__ == "__main__":
    clients_input, rounds_input, epochs_input, split_input, alpha_input, qtype_input, max_samples_input = get_user_setup()
    run_federated_simulation(clients_input, rounds_input, epochs_input, split_input, alpha_input, question_type=qtype_input, max_samples=max_samples_input)
