import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import time
import torch
import random
import warnings
import faiss
import numpy as np
import torch.nn as nn
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from sklearn.metrics import f1_score
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

# --- src/evaluation/metrics.py ---
class MedVQAEvaluator:
    def __init__(self):
        pass

    def evaluate_closed_ended(self, preds, refs):
        cleaned_preds = [str(p).lower().strip() for p in preds]
        cleaned_refs = [str(r).lower().strip() for r in refs]

        correct = 0
        mapped_preds = []
        for p, r in zip(cleaned_preds, cleaned_refs):
            if r in p or p in r:
                correct += 1
                mapped_preds.append(r)
            else:
                mapped_preds.append(p)

        accuracy = correct / len(refs) if refs else 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(cleaned_refs, mapped_preds, average='macro', zero_division=0)

        return {"Accuracy": accuracy, "F1-Score": f1}

    def evaluate_open_ended(self, preds, refs):
        cleaned_preds = [str(p).lower().strip() for p in preds]
        cleaned_refs = [str(r).lower().strip() for r in refs]

        bleu_scores = []
        rouge_l_scores = []

        for p, r in zip(cleaned_preds, cleaned_refs):
            p_tokens = p.split()
            r_tokens = r.split()

            if not r_tokens or not p_tokens:
                bleu_scores.append(0.0)
                rouge_l_scores.append(0.0)
                continue

            common_tokens = set(p_tokens).intersection(set(r_tokens))
            precision = len(common_tokens) / len(p_tokens)
            recall = len(common_tokens) / len(r_tokens)

            if precision + recall == 0:
                bleu, rouge = 0.0, 0.0
            else:
                brevity_penalty = 1.0 if len(p_tokens) > len(r_tokens) else np.exp(1 - len(r_tokens) / len(p_tokens))
                bleu = brevity_penalty * precision
                rouge = recall

            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge)

        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge_l = np.mean(rouge_l_scores) if rouge_l_scores else 0.0

        return {"BLEU": avg_bleu, "ROUGE-L": avg_rouge_l}

# --- src/rag_system/vector_db.py ---
class MedicalRetriever:
    def __init__(self, dataset_name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = dataset_name
        self.index = faiss.IndexFlatIP(1024)
        self.metadata = []
        
    def build_index(self, dataset, biomed_model, preprocess, tokenizer):
        embeds = []
        for item in dataset:
            img = item['image'].convert('RGB')
            q = str(item['question'])
            img_inp = preprocess(img).unsqueeze(0).to(self.device)
            txt_inp = tokenizer([q]).to(self.device)
            with torch.no_grad():
                img_f = biomed_model.encode_image(img_inp)
                txt_f = biomed_model.encode_text(txt_inp)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                comb = torch.cat([img_f, txt_f], dim=-1)
                comb = comb / comb.norm(dim=-1, keepdim=True)
                embeds.append(comb.cpu().numpy()[0])
        
        base_embeds = np.array(embeds, dtype=np.float32)
        self.index.add(base_embeds)
        
        for item in dataset:
            self.metadata.append({"question": item['question'], "answer": item['answer']})
            
        return base_embeds

    def compute_queries(self, query_dataset, biomed_model, preprocess, tokenizer):
        embeds = []
        for item in query_dataset:
            img = item['image'].convert('RGB')
            q = str(item['question'])
            img_inp = preprocess(img).unsqueeze(0).to(self.device)
            txt_inp = tokenizer([q]).to(self.device)
            with torch.no_grad():
                img_f = biomed_model.encode_image(img_inp)
                txt_f = biomed_model.encode_text(txt_inp)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                comb = torch.cat([img_f, txt_f], dim=-1)
                comb = comb / comb.norm(dim=-1, keepdim=True)
                embeds.append(comb.cpu().numpy()[0])
        return np.array(embeds, dtype=np.float32)

    def search_cases(self, query_embed, c=3, avoid_self_idx=None):
        search_k = c + (1 if avoid_self_idx is not None else 0)
        if search_k > self.index.ntotal:
            search_k = self.index.ntotal
        if search_k == 0:
            return []
            
        D, I = self.index.search(query_embed.reshape(1, -1), search_k)
        results = []
        for idx in I[0]:
            if idx == avoid_self_idx:
                continue
            results.append(self.metadata[idx])
            if len(results) == c:
                break
        return results

# --- src/models/qwen_slm.py (text-only variant) ---
class QwenMedVQA:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quantization_config = None
        if use_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if use_4bit else self.device,
            torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, question, context=""):
        messages = [
            {"role": "system", "content": "You are a precise medical AI assistant. Answer the question as briefly and accurately as possible based on the provided retrieved knowledge. For yes/no questions, output only 'yes' or 'no'. For open-ended questions, output only the direct answer word or phrase without extra explanations."},
            {"role": "user", "content": f"Retrieved Knowledge:\n{context}\n\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        prompt_len = inputs['input_ids'].shape[1]
        raw_ans = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        return raw_ans

# --- utilities ---
def is_closed_ended(item) -> bool:
    if 'answer_type' in item and item['answer_type'] is not None:
        return str(item['answer_type']).upper().strip() == 'CLOSED'
    ans = str(item.get('answer', '')).lower().strip()
    return ans in ['yes', 'no'] or len(ans.split()) <= 2

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_scores_for_json(c_scores, o_scores, question_type="all"):
    res = {}
    if question_type in ["all", "closed"]:
        res["Closed-Ended"] = {
            "Accuracy": round(c_scores.get('Accuracy', 0) * 100, 1),
            "F1-Score": round(c_scores.get('F1-Score', 0) * 100, 1)
        }
    if question_type in ["all", "open"]:
        res["Open-Ended"] = {
            "BLEU": round(o_scores.get('BLEU', 0) * 100, 1),
            "ROUGE-L": round(o_scores.get('ROUGE-L', 0) * 100, 1)
        }
    return res

def evaluate_dataset(shared_slm, dataset, rag_contexts, evaluator, question_type="all"):
    shared_slm.model.eval()
    closed_preds, closed_refs, open_preds, open_refs = [], [], [], []
    total = len(dataset)
    start_infer_time = time.time()
    
    for i, sample in enumerate(dataset):
        print(f"    Evaluating: {i+1}/{total} samples...", end="\r")
        question = sample['question']
        ground_truth = str(sample['answer']).lower()
        context = rag_contexts[i]
        
        sample_is_closed = (sample.get('answer_type', '').upper().strip() == 'CLOSED') if 'answer_type' in sample and sample['answer_type'] is not None else (ground_truth in ['yes', 'no'] or len(ground_truth.split()) <= 2)
        
        if question_type == "open" and sample_is_closed:
            continue
        elif question_type == "closed" and not sample_is_closed:
            continue
            
        pred = shared_slm.predict(question, context=context)
        pred_normalized = pred.strip().lower()
        
        # Normalize closed-ended predictions to exactly 'yes' or 'no'
        if sample_is_closed:
            first_word = pred_normalized.split()[0].strip('.,!?;:') if pred_normalized.split() else pred_normalized
            if first_word in ('yes', 'no'):
                pred_normalized = first_word
            elif 'yes' in pred_normalized:
                pred_normalized = 'yes'
            elif 'no' in pred_normalized:
                pred_normalized = 'no'
        
        if sample_is_closed:
            closed_preds.append(pred_normalized); closed_refs.append(ground_truth)
        else:
            open_preds.append(pred_normalized); open_refs.append(ground_truth)
            
    infer_time = round(time.time() - start_infer_time, 2)
    print(f"\nInference Time: {infer_time} seconds")
    
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs), infer_time

def run_centralized_training(epochs, question_type="all", max_samples=None):
    print(f"\nSTARTING CENTRALIZED TRAINING (TEXT-ONLY WITH RAG): {epochs} Epochs | Question Type = {question_type.upper()} | Max Samples = {max_samples if max_samples is not None else 'ALL'}")
    
    vqa_rad = load_dataset("mdwiratathya/SLAKE-vqa-english")
    vqa_rad_full = concatenate_datasets([vqa_rad["train"], vqa_rad["validation"], vqa_rad["test"]])
    
    eval_seed = 42
    random.seed(eval_seed)
    print(f"Using fixed dataset split seed: {eval_seed}")
    
    vqa_rad_full_shuffled = vqa_rad_full.shuffle(seed=eval_seed)
    eval_size = min(451, int(0.2 * len(vqa_rad_full_shuffled)))
    vqa_rad_eval = vqa_rad_full_shuffled.select(range(eval_size))
    vqa_rad_train_val = vqa_rad_full_shuffled.select(range(eval_size, len(vqa_rad_full_shuffled)))
    
    val_size = int(0.1 * len(vqa_rad_train_val))
    vqa_rad_val = vqa_rad_train_val.select(range(val_size))
    vqa_rad_train = vqa_rad_train_val.select(range(val_size, len(vqa_rad_train_val)))
    
    print(f"Dataset split: Train: {len(vqa_rad_train)}, Validation: {len(vqa_rad_val)}, Test: {len(vqa_rad_eval)}")
    
    evaluator = MedVQAEvaluator()

    # Filter train dataset by question type and max_samples
    if question_type in ["closed", "open"]:
        indices = []
        for idx, item in enumerate(vqa_rad_train):
            is_closed = is_closed_ended(item)
            if question_type == "closed" and is_closed:
                indices.append(idx)
            elif question_type == "open" and not is_closed:
                indices.append(idx)
        vqa_rad_train = vqa_rad_train.select(indices)
        print(f"[Dataset] Filtered train dataset to {question_type} questions: {len(indices)} samples.")
        
    if max_samples is not None and len(vqa_rad_train) > max_samples:
        vqa_rad_train = vqa_rad_train.select(range(max_samples))
        print(f"[Dataset] Limited train dataset to {len(vqa_rad_train)} samples (max_samples={max_samples}).")
        
    # Precompute RAG Contexts (BiomedCLIP)
    print("\n[RAG] Loading BiomedCLIP to precompute all FAISS indices and contexts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    biomed_model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    biomed_model, preprocess = open_clip.create_model_from_pretrained(biomed_model_name)
    biomed_tokenizer = open_clip.get_tokenizer(biomed_model_name)
    biomed_model.to(device).eval()

    # Build Central Index on Train set
    print("  Building FAISS index on Central Train Set...")
    retriever_train = MedicalRetriever("central_train")
    train_embeds = retriever_train.build_index(vqa_rad_train, biomed_model, preprocess, biomed_tokenizer)

    # Train RAG Contexts (avoid self)
    print("  Precomputing RAG contexts for Training Set...")
    train_rag_contexts = []
    for j in range(len(vqa_rad_train)):
        cases = retriever_train.search_cases(train_embeds[j], c=3, avoid_self_idx=j)
        ctx = "\n".join([f"- Question: {c['question']}\n  Answer: {c['answer']}" for c in cases])
        train_rag_contexts.append(ctx)

    # Global Test RAG Contexts
    print("  Precomputing RAG contexts for Test Evaluation...")
    eval_queries = retriever_train.compute_queries(vqa_rad_eval, biomed_model, preprocess, biomed_tokenizer)
    eval_rag_contexts = []
    for i in range(len(vqa_rad_eval)):
        cases = retriever_train.search_cases(eval_queries[i], c=3)
        ctx = "\n".join([f"- Question: {c['question']}\n  Answer: {c['answer']}" for c in cases])
        eval_rag_contexts.append(ctx)

    # Validation RAG Contexts
    print("  Precomputing RAG contexts for Validation...")
    val_queries = retriever_train.compute_queries(vqa_rad_val, biomed_model, preprocess, biomed_tokenizer)
    val_rag_contexts = []
    for i in range(len(vqa_rad_val)):
        cases = retriever_train.search_cases(val_queries[i], c=3)
        ctx = "\n".join([f"- Question: {c['question']}\n  Answer: {c['answer']}" for c in cases])
        val_rag_contexts.append(ctx)

    # Free BiomedCLIP memory
    del biomed_model
    torch.cuda.empty_cache()
    gc.collect()
    print("\n[RAG] BiomedCLIP freed from GPU. Proceeding to load Qwen LLM...")

    os.makedirs("./data", exist_ok=True)
    file_name = f"eval_results_centralized_{epochs}epochs_{question_type.upper()}_textonly_SLAKE.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Training_Type": "Centralized",
            "Epochs": epochs,
            "Question_Type": question_type.upper(),
            "Max_Samples": max_samples if max_samples is not None else "ALL",
            "Model_Type": "Centralized (Text-only Qwen) with RAG"
        },
        "Training_Stats": {},
        "Results": {
            "Centralized (SLAKE+RAG)": {}
        }
    }

    def save_current_progress(phase_name):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved progress '{phase_name}' to: {file_name}")

    print("\n>>> INITIALIZING SHARED QWEN TEXT ENGINE...")
    slm = QwenMedVQA(use_4bit=True)
    slm.model = prepare_model_for_kbit_training(slm.model)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    slm.model = get_peft_model(slm.model, lora_config)
    
    import torch.optim as optim
    
    checkpoint_path = f"./model_checkpoints/lora_centralized_{epochs}epochs_{question_type}_textonly_SLAKE.pt"
    os.makedirs("./model_checkpoints", exist_ok=True)

    model = slm.model
    tokenizer = slm.tokenizer
    device = slm.device

    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    total_train_time = 0.0
    final_avg_loss = 0.0
    start_train_time = time.time()

    if os.path.exists(checkpoint_path):
        print(f"\n[LOAD] Found existing trained weights checkpoint for SLAKE Centralized: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'weights' in checkpoint:
            model.load_state_dict(checkpoint['weights'], strict=False)
            best_loss = checkpoint.get('best_loss', float('inf'))
            final_avg_loss = best_loss if best_loss != float('inf') else 0.0
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Loaded saved weights from checkpoint, skipping training...")
    else:
        print(f"\n==================== EXPERIMENT: SLAKE Centralized Training ====================")
        model.train()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        accumulation_steps = 4
        indices = list(range(len(vqa_rad_train)))
        
        for epoch in range(epochs):
            random.shuffle(indices)
            total_loss = 0.0
            steps = 0
            optimizer.zero_grad()
            
            for i, idx in enumerate(indices):
                sample = vqa_rad_train[idx]
                question = sample['question']
                answer = str(sample['answer'])
                context = train_rag_contexts[idx]
                
                messages = [
                    {"role": "system", "content": "You are a precise medical AI assistant. Answer the question as briefly and accurately as possible based on the provided retrieved knowledge. For yes/no questions, output only 'yes' or 'no'. For open-ended questions, output only the direct answer word or phrase without extra explanations."},
                    {"role": "user", "content": f"Retrieved Knowledge:\n{context}\n\nQuestion: {question}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
                
                max_length = 512
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                
                inputs = {
                    "input_ids": torch.tensor([input_ids]).to(device),
                    "attention_mask": torch.tensor([[1]*len(input_ids)]).to(device),
                    "labels": torch.tensor([labels]).to(device)
                }
                
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        outputs = model(**inputs)
                        loss = outputs.loss / accumulation_steps
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(indices):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(indices):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                steps += 1
                
                if i % 10 == 0:
                    print(f"    Epoch {epoch+1}/{epochs} | Step {i}/{len(indices)} | Loss: {loss.item() * accumulation_steps:.4f}   ", end="\r")
                    
            print()
            avg_loss = total_loss / steps if steps > 0 else 0.0
            final_avg_loss = avg_loss
            print(f"  [SLAKE Centralized] Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            raw_weights = get_peft_model_state_dict(model)
            clean_weights = {}
            for name, param in raw_weights.items():
                if getattr(param, "device", None) and param.device.type == 'meta':
                    clean_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                else:
                    clean_weights[name] = param.clone().detach().cpu()
                    
            torch.save({
                'epoch': epoch + 1,
                'weights': clean_weights,
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }, checkpoint_path)
            print(f"  [SLAKE Centralized] Auto-saved Epoch {epoch+1} checkpoint.")

            if patience_counter >= patience:
                print(f"  [SLAKE Centralized] Early stopping triggered.")
                break
                
        total_train_time = round(time.time() - start_train_time, 2)
        
    results_dict["Training_Stats"]["SLAKE"] = {
        "Total_Train_Samples": len(vqa_rad_train),
        "Final_Average_Loss": round(final_avg_loss, 4) if isinstance(final_avg_loss, (int, float)) else final_avg_loss,
        "Training_Time_Seconds": total_train_time
    }
    
    print("\nEvaluating SLAKE Validation Set...")
    val_c, val_o, val_t = evaluate_dataset(slm, vqa_rad_val, val_rag_contexts, evaluator, question_type=question_type)
    results_dict["Results"]["Centralized (SLAKE+RAG)"]["SLAKE_Validation"] = format_scores_for_json(val_c, val_o, question_type=question_type)

    print("\nEvaluating SLAKE Test Set (Centralized+RAG)...")
    pv_c, pv_o, pv_t = evaluate_dataset(slm, vqa_rad_eval, eval_rag_contexts, evaluator, question_type=question_type)
    
    results_dict["Results"]["Centralized (SLAKE+RAG)"]["SLAKE_Test"] = format_scores_for_json(pv_c, pv_o, question_type=question_type)
    results_dict["Results"]["Centralized (SLAKE+RAG)"]["Inference_Time_Seconds"] = round(pv_t, 2)

    save_current_progress("All Experiments Completed")
    print(f"\nCOMPLETED! Evaluation metrics saved to: {json_path}")
    clear_memory()

def get_user_setup():
    while True:
        try:
            epochs = int(input("\n1. Enter training Epochs (e.g., 1, 2, 3 - 1 is recommended for speed/stability): "))
            if epochs >= 1: break
            else: print("At least 1 epoch is required!")
        except ValueError: print("Please enter a valid integer!")

    while True:
        question_type = input("2. Select question type to train on ('all', 'closed', 'open'): ").strip().lower()
        if question_type in ['all', 'closed', 'open']: break
        else: print("Only 'all', 'closed', or 'open' are accepted!")

    max_samples = None
    while True:
        max_samples_input = input("3. Enter max training samples per dataset (e.g., 1000, or press Enter for all): ").strip()
        if max_samples_input == "":
            max_samples = None
            break
        try:
            max_samples = int(max_samples_input)
            if max_samples >= 1: break
            else: print("Must be at least 1!")
        except ValueError:
            print("Please enter a valid integer or press Enter!")

    return epochs, question_type, max_samples

if __name__ == "__main__":
    epochs_input, qtype_input, max_samples_input = get_user_setup()
    run_centralized_training(epochs_input, question_type=qtype_input, max_samples=max_samples_input)
