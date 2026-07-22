from datetime import datetime
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
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from sklearn.metrics import f1_score
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

# --- SYSTEM PROMPT CHUẨN HÓA CẢI TIẾN (ĐÃ BỔ SUNG LỌC NHIỄU RAG) ---
SYSTEM_PROMPT = (
    "You are a precise medical AI assistant. Your task is to extract the exact answer based on the retrieved knowledge and visual evidence.\n"
    "Rules:\n"
    "- Pay attention to the similarity scores of retrieved cases. STRICTLY IGNORE or DISCARD any retrieved cases with low similarity or conflicting facts.\n"
    "- For CLOSED-ENDED (Yes/No) questions: Output MUST be strictly 'yes' or 'no'. Focus on visual evidence and rely ONLY on highly relevant cases.\n"
    "- For OPEN-ENDED questions: Synthesize ONLY from highly confident/relevant retrieved cases to provide the direct answer word or phrase without extra explanations."
)

# --- src/data_processing/data_splitter.py ---
class FederatedDataSplitter:
    @staticmethod
    def is_closed_ended(answer) -> bool:
        ans = str(answer).lower().strip()
        return ans in ['yes', 'no'] or len(ans.split()) <= 2

    def __init__(self, dataset: Dataset, num_clients: int = 2, seed: int = None, question_type: str = "all", max_samples: int = None):
        self.num_clients = num_clients
        self.seed = seed
        self.max_samples = max_samples
        
        filtered_dataset = dataset
        if question_type in ["closed", "open"]:
            indices = []
            for idx, item in enumerate(dataset):
                is_closed = self.is_closed_ended(item['answer'])
                if question_type == "closed" and is_closed:
                    indices.append(idx)
                elif question_type == "open" and not is_closed:
                    indices.append(idx)
            filtered_dataset = dataset.select(indices)
            print(f"[DataSplitter] Filtered dataset to {question_type} questions: {len(dataset)} -> {len(filtered_dataset)} samples.")
            
        self.dataset = filtered_dataset
        
    def split_iid(self):
        import random
        shuffle_seed = random.randint(0, 1000000) if self.seed is None else self.seed
        shuffled_dataset = self.dataset.shuffle(seed=shuffle_seed)
        split_size = len(shuffled_dataset) // self.num_clients
        print(f"[DataSplitter] IID split with seed {shuffle_seed}")
        
        client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = len(shuffled_dataset) if i == self.num_clients - 1 else (i + 1) * split_size
            client_ds = shuffled_dataset.select(range(start_idx, end_idx))
            
            if self.max_samples is not None:
                client_ds = client_ds.select(range(min(self.max_samples, len(client_ds))))
                print(f"[DataSplitter] Limited Hospital {i+1} dataset to {len(client_ds)} samples (max_samples={self.max_samples}).")
                
            client_datasets.append(client_ds)
        return client_datasets

    def split_non_iid(self, alpha: float = 0.5):
        import random
        dirichlet_seed = random.randint(0, 1000000) if self.seed is None else self.seed
        np.random.seed(dirichlet_seed)
        print(f"[DataSplitter] Non-IID split with seed {dirichlet_seed}")
        
        num_classes = 5
        answers = [str(ans).lower().strip() for ans in self.dataset['answer']]
        classes = np.array([hash(ans) % num_classes for ans in answers])
        class_indices = {c: np.where(classes == c)[0] for c in range(num_classes)}
        client_indices = [[] for _ in range(self.num_clients)]
        
        for c in range(num_classes):
            idx = class_indices[c]
            np.random.shuffle(idx)
            if len(idx) == 0:
                continue
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            idx_splits = np.split(idx, split_points)
            for i in range(self.num_clients):
                client_indices[i].extend(idx_splits[i].tolist())
                
        client_datasets = []
        for i, indices in enumerate(client_indices):
            np.random.shuffle(indices)
            if len(indices) == 0:
                indices = [np.random.randint(0, len(self.dataset))]
            client_ds = self.dataset.select(indices)
            
            if self.max_samples is not None:
                client_ds = client_ds.select(range(min(self.max_samples, len(client_ds))))
                
            client_datasets.append(client_ds)
            
        print(f"[DataSplitter] Non-IID splitted (alpha={alpha}).")
        for i, ds in enumerate(client_datasets):
            print(f"Hospital {i+1} receives: {len(ds)} photos (max_samples={self.max_samples}).")
        return client_datasets

# --- src/federated/server.py ---
class FederatedServer:
    def __init__(self):
        self.global_weights = None
        print("\n[Server] The central server started up")

    def aggregate_weights(self, client_weights_list, client_sizes=None):
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
            weighted_tensors = [t.to(torch.float32) * w for t, w in zip(tensors, weights)]
            avg_tensor = torch.stack(weighted_tensors).sum(dim=0)
            averaged_weights[key] = avg_tensor.to(tensors[0].dtype)
            
        self.global_weights = averaged_weights
        print("[Server] Done, new Global Model created")
        return self.global_weights

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
            if p == r:
                correct += 1
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

    def search_cases(self, query_embed, c=3, avoid_self_idx=None, score_threshold=0.65):
        search_k = c + (1 if avoid_self_idx is not None else 0)
        if search_k > self.index.ntotal:
            search_k = self.index.ntotal
        if search_k == 0:
            return []
            
        D, I = self.index.search(query_embed.reshape(1, -1), search_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == avoid_self_idx:
                continue
            # Chỉ giữ lại các case có Similarity Score >= score_threshold
            if score >= score_threshold:
                case_info = self.metadata[idx].copy()
                case_info['score'] = float(score)
                results.append(case_info)
            if len(results) == c:
                break
        return results

# --- src/models/qwen_slm.py ---
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
            {"role": "system", "content": SYSTEM_PROMPT},
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

# --- Virtual Client ---
class VirtualClient:
    def __init__(self, client_id, local_dataset, rag_contexts, initial_weights):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.rag_contexts = rag_contexts
        self.lora_weights = {k: v.clone() for k, v in initial_weights.items()}

    def train_local(self, shared_slm, epochs=1):
        import torch.optim as optim
        
        print(f"  [{self.client_id}] Training locally for {epochs} epoch(s)...")
        model = shared_slm.model
        tokenizer = shared_slm.tokenizer
        device = shared_slm.device
        
        model.load_state_dict(self.lora_weights, strict=False)
        model.train()
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        total_loss = 0.0
        steps = 0
        accumulation_steps = 4
        
        indices = list(range(len(self.local_dataset)))
        
        for epoch in range(epochs):
            random.shuffle(indices)
            epoch_indices = indices
            optimizer.zero_grad()
            
            for i, idx in enumerate(epoch_indices):
                sample = self.local_dataset[idx]
                question = sample['question']
                answer = str(sample['answer'])
                context = self.rag_contexts[idx]
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
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
                    
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_indices):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_indices):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                steps += 1
                
                if i % 10 == 0:
                    print(f"    [{self.client_id}] Epoch {epoch+1}/{epochs} | Step {i}/{len(epoch_indices)} | Loss: {loss.item() * accumulation_steps:.4f}   ", end="\r")
                    
        print() 
        avg_loss = total_loss / steps if steps > 0 else 0.0
        
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
        
        is_closed = ground_truth in ['yes', 'no'] or len(ground_truth.split()) <= 2
        
        if question_type == "open" and is_closed:
            continue
        elif question_type == "closed" and not is_closed:
            continue
            
        pred = shared_slm.predict(question, context=context)
        pred_normalized = pred.strip().lower()
        
        if is_closed:
            first_word = pred_normalized.split()[0].strip('.,!?;:') if pred_normalized.split() else pred_normalized
            if first_word in ('yes', 'no'):
                pred_normalized = first_word
            elif 'yes' in pred_normalized:
                pred_normalized = 'yes'
            elif 'no' in pred_normalized:
                pred_normalized = 'no'
        
        if is_closed:
            closed_preds.append(pred_normalized); closed_refs.append(ground_truth)
        else:
            open_preds.append(pred_normalized); open_refs.append(ground_truth)
            
    infer_time = round(time.time() - start_infer_time, 2)
    print(f"\nInference Time: {infer_time} seconds")
    
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs), infer_time

def format_rag_context(cases):
    if not cases:
        return "No highly relevant cases found."
    return "\n".join([f"- Case (Similarity: {c['score']:.2f}): Question: {c['question']} | Answer: {c['answer']}" for c in cases])

def run_federated_simulation(num_clients, num_rounds, epochs, split_type, alpha, question_type="all", max_samples=None, results_dict=None):
    print(f"\nSTARTING DECOUPLED FL (TEXT-ONLY WITH RAG): {num_clients} Clients | {num_rounds} Rounds | {epochs} Epochs | {split_type.upper()} | Alpha = {alpha if split_type == 'non-iid' else 'NA'} | Question Type = {question_type.upper()} | Max Samples = {max_samples if max_samples is not None else 'ALL'}")
    
    from datasets import concatenate_datasets
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    vqa_rad_full = concatenate_datasets([vqa_rad["train"], vqa_rad["test"]])
    
    eval_seed = 42
    random.seed(eval_seed)
    
    vqa_rad_full_shuffled = vqa_rad_full.shuffle(seed=eval_seed)
    eval_size = 451
    vqa_rad_eval = vqa_rad_full_shuffled.select(range(eval_size))
    vqa_rad_train_val = vqa_rad_full_shuffled.select(range(eval_size, len(vqa_rad_full_shuffled)))
    
    val_size = int(0.1 * len(vqa_rad_train_val))
    vqa_rad_val = vqa_rad_train_val.select(range(val_size))
    vqa_rad_train = vqa_rad_train_val.select(range(val_size, len(vqa_rad_train_val)))
    
    evaluator = MedVQAEvaluator()
    server = FederatedServer()

    splitter_seed_rad = 42
    splitter_rad = FederatedDataSplitter(vqa_rad_train, num_clients=num_clients, seed=splitter_seed_rad, question_type=question_type, max_samples=max_samples)
    if split_type == 'iid':
        client_datasets_rad = splitter_rad.split_iid()
    else:
        client_datasets_rad = splitter_rad.split_non_iid(alpha=alpha)
        
    total_samples_rad = sum(len(ds) for ds in client_datasets_rad)
    contributions_rad = {f"Hospital_{i+1}": round((len(ds) / total_samples_rad) * 100, 2) for i, ds in enumerate(client_datasets_rad)}
    
    print("\n[RAG] Loading BiomedCLIP to precompute all FAISS indices and contexts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    biomed_model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    biomed_model, preprocess = open_clip.create_model_from_pretrained(biomed_model_name)
    biomed_tokenizer = open_clip.get_tokenizer(biomed_model_name)
    biomed_model.to(device).eval()

    retriever_global = MedicalRetriever("global")
    retriever_global.build_index(vqa_rad_train, biomed_model, preprocess, biomed_tokenizer)
    eval_queries = retriever_global.compute_queries(vqa_rad_eval, biomed_model, preprocess, biomed_tokenizer)
    eval_rag_contexts = []
    for i in range(len(vqa_rad_eval)):
        cases = retriever_global.search_cases(eval_queries[i], c=3, score_threshold=0.65)
        eval_rag_contexts.append(format_rag_context(cases))

    val_queries = retriever_global.compute_queries(vqa_rad_val, biomed_model, preprocess, biomed_tokenizer)
    val_rag_contexts = []
    for i in range(len(vqa_rad_val)):
        cases = retriever_global.search_cases(val_queries[i], c=3, score_threshold=0.65)
        val_rag_contexts.append(format_rag_context(cases))

    client_rag_contexts = []
    for i, ds in enumerate(client_datasets_rad):
        retriever_local = MedicalRetriever(f"client_{i}")
        local_embeds = retriever_local.build_index(ds, biomed_model, preprocess, biomed_tokenizer)
        rag_ctx = []
        for j in range(len(ds)):
            cases = retriever_local.search_cases(local_embeds[j], c=3, avoid_self_idx=j, score_threshold=0.65)
            rag_ctx.append(format_rag_context(cases))
        client_rag_contexts.append(rag_ctx)

    del biomed_model
    clear_memory()

    os.makedirs("./data", exist_ok=True)
    json_path = "./data/eval_results_k_clients_experiment.json"

    exp_key = f"{num_clients}_clients_{split_type.upper()}"
    if results_dict is None:
        results_dict = {}

    results_dict[exp_key] = {
        "Experiment_Config": {
            "Num_Clients": num_clients,
            "Max_Rounds_Configured": num_rounds,
            "Local_Epochs": epochs,
            "Split_Type": split_type.upper(),
            "Alpha": alpha if split_type == 'non-iid' else "NA",
            "Model_Type": "Decoupled FL (Qwen2.5-1.5B) with Score-Filtered RAG"
        },
        "Training_Stats": {},
        "Results": {
            "Proposed (Fed+RAG)": {}
        }
    }

    def save_current_progress():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved progress for [{exp_key}] to: {json_path}")

    print("\n>>> INITIALIZING SHARED QWEN2.5-1.5B TEXT ENGINE...")
    shared_slm = QwenMedVQA(model_id="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=True)
    shared_slm.model = prepare_model_for_kbit_training(shared_slm.model)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    shared_slm.model = get_peft_model(shared_slm.model, lora_config)
    
    initial_lora_weights = {}
    for k, v in get_peft_model_state_dict(shared_slm.model).items():
        if getattr(v, "device", None) and v.device.type == 'meta':
            initial_lora_weights[k] = torch.zeros(v.shape, dtype=v.dtype, device='cpu')
        else:
            initial_lora_weights[k] = v.clone().cpu()

    shared_slm.model.load_state_dict(initial_lora_weights, strict=False)
    
    alpha_str = str(alpha) if split_type == 'non-iid' else "NA"
    checkpoint_rad = f"./model_checkpoints/lora_qwen1.5b_rad_{num_clients}clients_{num_rounds}rounds_{epochs}epochs_{split_type}_a{alpha_str}_textonly.pt"
    os.makedirs("./model_checkpoints", exist_ok=True)

    global_weights = None
    start_round = 1
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    actual_rounds = 0
    final_loss = 0.0
    total_train_time_rad = 0.0
    start_train_time = time.time()

    if os.path.exists(checkpoint_rad):
        print(f"\n[LOAD] Found existing trained weights checkpoint: {checkpoint_rad}")
        checkpoint = torch.load(checkpoint_rad, map_location='cpu')
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

    clients = []
    if start_round <= num_rounds:
        for i in range(num_clients):
            initial_client_weights = global_weights if global_weights is not None else initial_lora_weights
            clients.append(VirtualClient(f"Hospital_{i+1}", client_datasets_rad[i], client_rag_contexts[i], initial_client_weights))
            
        for round_num in range(start_round, num_rounds + 1):
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
                
            torch.save({
                'round': round_num,
                'weights': global_weights,
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }, checkpoint_rad)

            if patience_counter >= patience:
                print(f"  [VQA-RAD] Early stopping triggered.")
                break
                
        total_train_time_rad = round(time.time() - start_train_time, 2)
        
    results_dict[exp_key]["Training_Stats"]["VQA-RAD"] = {
        "Client_Data_Contributions_Percent": contributions_rad,
        "Actual_Rounds_Run": actual_rounds,
        "Final_Average_Loss": round(final_loss, 4) if isinstance(final_loss, (int, float)) else final_loss,
        "Training_Time_Seconds": total_train_time_rad
    }
    
    shared_slm.model.load_state_dict(global_weights, strict=False)
    print("\nEvaluating VQA-RAD Validation Set...")
    val_c, val_o, val_t = evaluate_dataset(shared_slm, vqa_rad_val, val_rag_contexts, evaluator, question_type=question_type)
    results_dict[exp_key]["Results"]["Proposed (Fed+RAG)"]["VQA-RAD_Validation"] = format_scores_for_json(val_c, val_o, question_type=question_type)

    print("\nEvaluating VQA-RAD Test Set (Decoupled Fed+RAG)...")
    pv_c, pv_o, pv_t = evaluate_dataset(shared_slm, vqa_rad_eval, eval_rag_contexts, evaluator, question_type=question_type)
    
    results_dict[exp_key]["Results"]["Proposed (Fed+RAG)"]["VQA-RAD_Test"] = format_scores_for_json(pv_c, pv_o, question_type=question_type)
    results_dict[exp_key]["Results"]["Proposed (Fed+RAG)"]["Inference_Time_Seconds"] = round(pv_t, 2)

    save_current_progress()

    if 'clients' in locals():
        del clients
    del client_datasets_rad
    del shared_slm
    clear_memory()
    return results_dict

def main():
    k_list = [2, 3, 5, 10]
    num_rounds = 10
    epochs = 1
    alpha = 0.5
    question_type = "all"
    max_samples = None
    split_modes = ['iid', 'non-iid']

    json_path = "./data/eval_results_k_clients_experiment.json"
    results_dict = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results_dict = json.load(f)
            print(f"[LOAD] Loaded existing results from {json_path}")
        except Exception as e:
            print(f"[WARN] Failed to load existing json: {e}")

    for k in k_list:
        for split_mode in split_modes:
            print("\n" + "="*80)
            print(f"   STARTING EXPERIMENT LOOP: K = {k} CLIENTS | SPLIT = {split_mode.upper()}")
            print("="*80)
            results_dict = run_federated_simulation(
                num_clients=k,
                num_rounds=num_rounds,
                epochs=epochs,
                split_type=split_mode,
                alpha=alpha,
                question_type=question_type,
                max_samples=max_samples,
                results_dict=results_dict
            )

if __name__ == "__main__":
    main()