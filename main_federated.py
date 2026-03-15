import json
import os
import gc
import torch
from datasets import load_from_disk
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
    
    for i, sample in enumerate(test_samples):
        print(f"Test samples: {i+1}/{total}...", end="\r")
        
        question = sample['question']
        ground_truth = str(sample['answer']).lower()
        image = sample['image']
        
        if retriever:
            similar_cases = retriever.search_similar_cases(image, k=2)
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
            
    print()
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs)


class VirtualClient:
    def __init__(self, client_id, local_dataset, initial_weights):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.lora_weights = {k: v.clone() for k, v in initial_weights.items()}

    def train_local(self, shared_model, epochs=1):
        print(f"  [{self.client_id}] Local training...")
        shared_model.load_state_dict(self.lora_weights, strict=False)
        shared_model.train()
        
        raw_weights = get_peft_model_state_dict(shared_model)
        self.lora_weights = {}
        for name, param in raw_weights.items():
            if param.device.type == 'meta':
                self.lora_weights[name] = torch.zeros_like(param, device='cpu')
            else:
                self.lora_weights[name] = param.clone().detach().cpu()
        return self.lora_weights


def run_federated_simulation():
    print("Experiment started")
    
    # Prepare data
    vqa_rad_data = load_from_disk("./data/vqa_rad_subset_50")['train']
    path_vqa_data = load_from_disk("./data/path_vqa_subset_100")['train']
    splitter = FederatedDataSplitter(vqa_rad_data, num_clients=2)
    client_datasets = splitter.split_iid()
    evaluator = MedVQAEvaluator()
    server = FederatedServer()

    print("\nBuild Vector RAG (FAISS)...")
    retriever_vr = MedicalRetriever(); retriever_vr.build_index_from_dataset(vqa_rad_data)
    retriever_pv = MedicalRetriever(); retriever_pv.build_index_from_dataset(path_vqa_data)

    os.makedirs("./data", exist_ok=True)
    json_path = "./data/evaluation_results.json"
    results_dict = {}

    def save_current_progress(phase_name):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"[Auto-Save] '{phase_name}' saved to JSON file")

    print("\n Initialize QWEN2-VL (Shared Engine)...")
    shared_slm = QwenMedVQA(use_4bit=True)
    shared_slm.model = prepare_model_for_kbit_training(shared_slm.model)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    shared_slm.model = get_peft_model(shared_slm.model, lora_config)

    initial_lora_weights = {k: v.clone().cpu() for k, v in get_peft_model_state_dict(shared_slm.model).items()}

    # Centralized + RAG
    print("\nCentralized + RAG")
    centralized_client = VirtualClient("Centralized", vqa_rad_data, initial_lora_weights)
    centralized_weights = centralized_client.train_local(shared_slm.model, epochs=1)
    
    shared_slm.model.load_state_dict(centralized_weights, strict=False)
    
    vr_c, vr_o = evaluate_dataset(shared_slm, vqa_rad_data, evaluator, retriever_vr)
    pv_c, pv_o = evaluate_dataset(shared_slm, path_vqa_data, evaluator, retriever_pv)
    
    results_dict["Centralized+RAG"] = {"VQA-RAD": format_scores_for_json(vr_c, vr_o), "PathVQA": format_scores_for_json(pv_c, pv_o)}
    save_current_progress("Centralized+RAG")

    # Federated Learning
    print("\nFederated Learning")
    client1 = VirtualClient("Client_A", client_datasets[0], initial_lora_weights)
    client2 = VirtualClient("Client_B", client_datasets[1], initial_lora_weights)
    clients = [client1, client2]
    
    global_weights = None
    for round_num in range(1, 2):
        print(f"Round {round_num}: Getting knowledge...")
        client_weights_list = [client.train_local(shared_slm.model) for client in clients]
        global_weights = server.aggregate_weights(client_weights_list)
        for client in clients:
            client.lora_weights = {k: v.clone() for k, v in global_weights.items()}

    # FED-SLM (No RAG)
    print("\nFed-SLM (No RAG)")
    shared_slm.model.load_state_dict(global_weights, strict=False)
    
    vr_c, vr_o = evaluate_dataset(shared_slm, vqa_rad_data, evaluator, None)
    pv_c, pv_o = evaluate_dataset(shared_slm, path_vqa_data, evaluator, None)
    
    results_dict["Fed-SLM (No RAG)"] = {"VQA-RAD": format_scores_for_json(vr_c, vr_o), "PathVQA": format_scores_for_json(pv_c, pv_o)}
    save_current_progress("Fed-SLM (No RAG)")

    # Proposed (FED + RAG)
    print("\nProposed (Fed + RAG)")
    vr_c, vr_o = evaluate_dataset(shared_slm, vqa_rad_data, evaluator, retriever_vr)
    pv_c, pv_o = evaluate_dataset(shared_slm, path_vqa_data, evaluator, retriever_pv)
    
    results_dict["Proposed (Fed+RAG)"] = {"VQA-RAD": format_scores_for_json(vr_c, vr_o), "PathVQA": format_scores_for_json(pv_c, pv_o)}
    save_current_progress("Proposed (Fed+RAG)")

    print(f"\nDone! Saved to JSON file at: {json_path}")

if __name__ == "__main__":
    run_federated_simulation()