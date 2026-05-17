import json
import os
import gc
import time
import torch
import random
from datasets import load_dataset, concatenate_datasets
from src.data_processing.data_splitter import FederatedDataSplitter
from src.federated.server import FederatedServer
from src.evaluation.metrics import MedVQAEvaluator
from src.rag_system.vector_db import MedicalRetriever
from src.models.qwen_slm import QwenMedVQA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict

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
            similar_cases = retriever.search_similar_cases(image, k=10)
            context_text = "Here are some similar reference cases:\n" if similar_cases else ""
            for j, case in enumerate(similar_cases):
                context_text += f"- Ref {j+1}: Q: '{case['question']}' -> A: '{case['answer']}'\n"
            augmented_question = f"{context_text}\nNow, please answer this new Question: {question}"
            pred = shared_slm.predict(image, augmented_question)
        else:
            pred = shared_slm.predict(image, question)
            
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

def run_centralized_simulation(num_clients, num_rounds, epochs, split_type, alpha):
    print(f"\nSTARTING CENTRALIZED EXPERIMENT: {num_clients} Clients | {epochs} Epochs | {split_type.upper()} | Alpha = {alpha if split_type == 'non-iid' else 'NA'}")
    
    # Load full datasets
    # Load Train and Test splits separately
    # Load directly from Hugging Face Hub
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    
    vqa_rad_train = vqa_rad["train"]
    path_vqa_train = path_vqa["train"]
    vqa_rad_test = vqa_rad["test"]
    path_vqa_test = path_vqa["test"]
    
    # Use time-based random sampling for evaluation only
    random.seed(int(time.time()))
    eval_seed = random.randint(0, 1000000)
    
    print(f"Using random evaluation seed: {eval_seed}")
    
    # Create truly random evaluation sets (100 images total)
    # 1. Prepare evaluation sets (using official Test data)
    vqa_rad_shuffled = vqa_rad_test.shuffle(seed=eval_seed)
    path_vqa_shuffled = path_vqa_test.shuffle(seed=eval_seed+1)
    
    eval_size = 100
    vqa_rad_eval = vqa_rad_shuffled.select(range(min(eval_size, len(vqa_rad_test))))
    path_vqa_eval = path_vqa_shuffled.select(range(min(eval_size, len(path_vqa_test))))
    
    print(f"Evaluating with {len(vqa_rad_eval)} VQA-RAD and {len(path_vqa_eval)} PathVQA images (from official TEST set)")
    
    # 2. Combine BOTH datasets for Centralized Training
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets([vqa_rad_train, path_vqa_train])
    
    # 3. Use combined dataset for splitting/gathering
    splitter_seed = random.randint(0, 1000000)
    splitter = FederatedDataSplitter(combined_train, num_clients=num_clients, seed=splitter_seed)
    print(f"Using splitter seed: {splitter_seed}")
    
    if split_type == 'iid':
        print(f"\n[1/3] Splitting data using IID for {num_clients} Hospitals...")
        client_datasets = splitter.split_iid()
        alpha_str = "NA"
    else:
        print(f"\n[1/3] Splitting data using Non-IID (alpha={alpha}) for {num_clients} Hospitals...")
        client_datasets = splitter.split_non_iid(alpha=alpha)
        alpha_str = str(alpha)

    total_samples = sum(len(ds) for ds in client_datasets)
    client_contributions = {
        f"Hospital_{i+1}": round((len(ds) / total_samples) * 100, 2) 
        for i, ds in enumerate(client_datasets)
    }

    # GATHER ALL CLIENT DATA TOGETHER (CENTRALIZED TRAINING)
    print(f"\n[2/3] Gathering all data from {num_clients} Hospitals for Centralized Training...")
    print(f"Original data distribution: {client_contributions}")
    
    # Combine all client datasets
    vqa_rad_centralized = concatenate_datasets(client_datasets)
    print(f"Combined dataset size: {len(vqa_rad_centralized)} samples")
    print("Centralized training on ALL data together (Upper Bound Performance)")

    evaluator = MedVQAEvaluator()

    print("\n[3/3] Building RAG Vector Database (FAISS)...")
    # Use subset for RAG to avoid memory issues
    # Build RAG pool from Training data only
    vqa_rad_rag_pool = vqa_rad_train.shuffle(seed=eval_seed+2).select(range(min(2000, len(vqa_rad_train))))
    path_vqa_rag_pool = path_vqa_train.shuffle(seed=eval_seed+3).select(range(min(2000, len(path_vqa_train))))
    
    retriever_vr = MedicalRetriever(); retriever_vr.build_index_from_dataset(vqa_rad_rag_pool)
    retriever_pv = MedicalRetriever(); retriever_pv.build_index_from_dataset(path_vqa_rag_pool)
    
    print(f"Built RAG index with {len(vqa_rad_rag_pool)} VQA-RAD and {len(path_vqa_rag_pool)} PathVQA images")

    os.makedirs("./data", exist_ok=True)
    file_name = f"eval_results_centralized_{num_clients}clients_{epochs}epochs_{split_type.upper()}_a{alpha_str}.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Training_Type": "Centralized",
            "Num_Clients": num_clients,
            "Local_Epochs": epochs,
            "Split_Type": split_type.upper(),
            "Alpha": alpha if split_type == 'non-iid' else "NA",
            "Data_Distribution": client_contributions
        },
        "Training_Stats": {
            "Total_Samples": len(vqa_rad_centralized),
            "Training_Time_Seconds": 0.0
        },
        "Results": {}
    }

    def save_current_progress(phase_name):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved progress '{phase_name}' to: {file_name}")

    print("\n>>> INITIALIZING CENTRALIZED QWEN2-VL ENGINE...")
    centralized_slm = QwenMedVQA(use_4bit=True)
    centralized_slm.model = prepare_model_for_kbit_training(centralized_slm.model)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    centralized_slm.model = get_peft_model(centralized_slm.model, lora_config)
    
    print(f"\n>>> CENTRALIZED TRAINING on {len(vqa_rad_centralized)} samples...")
    centralized_slm.model.train()
    
    start_train_time = time.time() # START TRAINING TIMER
    
    # Actual PyTorch Training Loop
    import torch.optim as optim
    from qwen_vl_utils import process_vision_info
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, centralized_slm.model.parameters()), lr=2e-5)
    
    total_loss = 0.0
    steps = 0
    indices = list(range(len(vqa_rad_centralized)))
    
    for epoch in range(epochs):
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            sample = vqa_rad_centralized[idx]
            question = sample['question']
            answer = str(sample['answer']).lower()
            
            img_obj = centralized_slm._preprocess_image(sample['image'])
            
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
            
            text = centralized_slm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = centralized_slm.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(centralized_slm.device)
            
            inputs['labels'] = inputs['input_ids'].clone()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = centralized_slm.model(**inputs)
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if i % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Step {i}/{len(indices)} | Loss: {loss.item():.4f}   ", end="\r")
                
    print()  # newline after progress
    
    final_loss = total_loss / steps if steps > 0 else 0.0
    
    raw_weights = get_peft_model_state_dict(centralized_slm.model)
    centralized_weights = {}
    for name, param in raw_weights.items():
        if getattr(param, "device", None) and param.device.type == 'meta':
            centralized_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
        else:
            centralized_weights[name] = param.clone().detach().cpu()
    
    end_train_time = time.time() # END TRAINING TIMER
    total_train_time = round(end_train_time - start_train_time, 2)
    print(f"\nCentralized Training Time: {total_train_time} seconds")
    print(f"Final Centralized Loss: {final_loss:.4f}")

    results_dict["Training_Stats"]["Training_Time_Seconds"] = total_train_time
    results_dict["Training_Stats"]["Final_Average_Loss"] = round(final_loss, 4)
    save_current_progress("Centralized Training Completed")

    centralized_slm.model.load_state_dict(centralized_weights, strict=False)
    
    print("\n>>> CENTRALIZED + RAG EVALUATION (Upper Bound)")
    vr_c, vr_o, vr_t_rag = evaluate_dataset(centralized_slm, vqa_rad_eval, evaluator, retriever_vr)
    pv_c, pv_o, pv_t_rag = evaluate_dataset(centralized_slm, path_vqa_eval, evaluator, retriever_pv)
    results_dict["Results"]["Centralized + RAG"] = {
        "VQA-RAD": format_scores_for_json(vr_c, vr_o), 
        "PathVQA": format_scores_for_json(pv_c, pv_o),
        "Inference_Time_Seconds": round(vr_t_rag + pv_t_rag, 2)
    }
    save_current_progress("Centralized + RAG")

    print(f"\nCOMPLETED! Centralized evaluation metrics saved to: {json_path}")

def get_user_setup():
    
    while True:
        try:
            num_clients = int(input("\n1. Enter number of participating Hospitals (e.g., 2, 3, 5): "))
            if num_clients >= 2: break
            else: print("At least 2 Hospitals are required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        try:
            epochs = int(input("2. Enter training Epochs (e.g., 5, 10, 20): "))
            if epochs >= 1: break
            else: print("At least 1 epoch is required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        split_type = input("3. Select data splitting mode ('iid' or 'non-iid'): ").strip().lower()
        if split_type in ['iid', 'non-iid']: break
        else: print("Only 'iid' or 'non-iid' are accepted!")

    alpha = 0.5 
    if split_type == 'non-iid':
        while True:
            try:
                alpha = float(input("4. Enter Alpha coefficient (e.g., 0.1 for extreme non-IID, 0.5 for moderate): "))
                if alpha > 0: break
                else: print("Alpha must be greater than 0!")
            except ValueError: print("Please enter a valid float number!")

    return num_clients, 0, epochs, split_type, alpha  # num_rounds=0 for centralized

if __name__ == "__main__":
    clients_input, rounds_input, epochs_input, split_input, alpha_input = get_user_setup()
    run_centralized_simulation(clients_input, rounds_input, epochs_input, split_input, alpha_input)
