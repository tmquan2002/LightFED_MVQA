import os
# Fix duplicate OpenMP runtime library error on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import time
import torch
import random
import faiss
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import open_clip

# Download NLTK resources silently if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MedVQAEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def compute_exact_match(self, prediction, ground_truth):
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def compute_f1(self, prediction, ground_truth):
        pred_tokens = prediction.strip().lower().split()
        gt_tokens = ground_truth.strip().lower().split()
        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    def compute_bleu(self, prediction, ground_truth):
        ref = [ground_truth.strip().lower().split()]
        hyp = prediction.strip().lower().split()
        smooth = SmoothingFunction().method1
        return sentence_bleu(ref, hyp, smoothing_function=smooth)

    def compute_rouge_l(self, prediction, ground_truth):
        scores = self.scorer.score(ground_truth.strip().lower(), prediction.strip().lower())
        return scores['rougeL'].fmeasure

    def evaluate(self, predictions, ground_truths):
        ems, f1s, bleus, rouges = [], [], [], []
        for p, r in zip(predictions, ground_truths):
            ems.append(self.compute_exact_match(p, r))
            f1s.append(self.compute_f1(p, r))
            bleus.append(self.compute_bleu(p, r))
            rouges.append(self.compute_rouge_l(p, r))
            
        return {
            "EM": np.mean(ems),
            "F1-Score": np.mean(f1s),
            "BLEU": np.mean(bleus),
            "ROUGE-L": np.mean(rouges)
        }

def precompute_embeddings(hf_dataset, biomed_model, preprocess, tokenizer, device):
    embeddings = []
    
    for idx in tqdm(range(len(hf_dataset)), desc="Computing Embeddings"):
        item = hf_dataset[idx]
        image = item['image'].convert('RGB')
        question = str(item['question'])
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = tokenizer([question]).to(device)
        
        with torch.no_grad():
            image_features = biomed_model.encode_image(image_input)
            text_features = biomed_model.encode_text(text_input)
            
            # L2-Normalize separately
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Combine
            combined = torch.cat([image_features, text_features], dim=-1)
            # L2-Normalize the concatenated vector for Cosine Similarity
            combined = combined / combined.norm(dim=-1, keepdim=True)
            embeddings.append(combined.cpu().numpy()[0])
            
    return np.array(embeddings, dtype=np.float32)

class MedVQADataset(Dataset):
    def __init__(self, hf_dataset, retrieved_contexts, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.retrieved_contexts = retrieved_contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        context = self.retrieved_contexts[idx]
        answer = str(item['answer'])
        
        prompt = f"Question:\n{question}\n\nRetrieved Knowledge:\n{context}\n\nAnswer:\n"
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def get_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        max_len = max(len(x) for x in input_ids)
        
        padded_input_ids, padded_labels, padded_attention_mask = [], [], []
        
        for ids, labs, mask in zip(input_ids, labels, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'labels': torch.tensor(padded_labels),
            'attention_mask': torch.tensor(padded_attention_mask)
        }
    return collate_fn

def format_retrieved_knowledge(dataset, indices):
    docs = []
    for idx in indices:
        item = dataset[idx]
        docs.append(f"- Question: {item['question']}\n  Answer: {item['answer']}")
    return "\n".join(docs)

def evaluate_model(model, tokenizer, dataset, contexts, evaluator, device):
    model.eval()
    predictions = []
    ground_truths = []
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset[idx]
        question = item['question']
        context = contexts[idx]
        gt = str(item['answer'])
        ground_truths.append(gt)
        
        prompt = f"Question:\n{question}\n\nRetrieved Knowledge:\n{context}\n\nAnswer:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1
            )
            
        prompt_len = inputs['input_ids'].shape[1]
        gen_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(gen_text)
        
    metrics = evaluator.evaluate(predictions, ground_truths)
    return metrics, predictions

def run_centralized_training():
    epochs = 5
    batch_size = 4
    k = 3
    lr = 2e-4
    gradient_accumulation_steps = 2
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(42)
    
    print("Loading PathVQA Dataset...")
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    train_ds = path_vqa["train"]
    val_ds = path_vqa["validation"]
    test_ds = path_vqa["test"]
    
    print("Step 1: Precomputing Embeddings...")
    biomed_model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomed_model, preprocess = open_clip.create_model_from_pretrained(biomed_model_id)
    biomed_tokenizer = open_clip.get_tokenizer(biomed_model_id)
    biomed_model.to(device)
    biomed_model.eval()
    
    # Check if we can load from cache
    os.makedirs("./data/embeddings", exist_ok=True)
    cache_train = "./data/embeddings/train_embeds.npy"
    cache_val = "./data/embeddings/val_embeds.npy"
    cache_test = "./data/embeddings/test_embeds.npy"
    
    if os.path.exists(cache_train) and os.path.exists(cache_val) and os.path.exists(cache_test):
        print("Found cached embeddings. Loading...")
        train_embeds = np.load(cache_train)
        val_embeds = np.load(cache_val)
        test_embeds = np.load(cache_test)
    else:
        train_embeds = precompute_embeddings(train_ds, biomed_model, preprocess, biomed_tokenizer, device)
        val_embeds = precompute_embeddings(val_ds, biomed_model, preprocess, biomed_tokenizer, device)
        test_embeds = precompute_embeddings(test_ds, biomed_model, preprocess, biomed_tokenizer, device)
        
        np.save(cache_train, train_embeds)
        np.save(cache_val, val_embeds)
        np.save(cache_test, test_embeds)
    
    # Clean memory
    del biomed_model
    torch.cuda.empty_cache()
    gc.collect()
    print("BiomedCLIP memory cleared.")
    
    print("Step 2: Building FAISS Index & Retrieval...")
    index = faiss.IndexFlatIP(1024)
    index.add(train_embeds)
    
    # Train context (avoid self)
    train_contexts = []
    D_train, I_train = index.search(train_embeds, k + 1)
    for idx in range(len(train_ds)):
        retrieved = []
        for match_idx in I_train[idx]:
            if match_idx == idx: continue
            retrieved.append(match_idx)
            if len(retrieved) == k: break
        if len(retrieved) < k: # Fallback
            for match_idx in I_train[idx]:
                if match_idx not in retrieved:
                    retrieved.append(match_idx)
                if len(retrieved) == k: break
        train_contexts.append(format_retrieved_knowledge(train_ds, retrieved))
        
    # Val/Test context
    val_contexts = []
    _, I_val = index.search(val_embeds, k)
    for idx in range(len(val_ds)):
        val_contexts.append(format_retrieved_knowledge(train_ds, I_val[idx]))
        
    test_contexts = []
    _, I_test = index.search(test_embeds, k)
    for idx in range(len(test_ds)):
        test_contexts.append(format_retrieved_knowledge(train_ds, I_test[idx]))
        
    print("Step 3: Setup LLM...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    print("Step 4: Dataloaders & Optimizer...")
    train_dataset = MedVQADataset(train_ds, train_contexts, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=get_collate_fn(tokenizer)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    evaluator = MedVQAEvaluator()
    best_em = -1.0
    
    print("Step 5: Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            
        print(f"Epoch {epoch} Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Validation Eval
        val_metrics, _ = evaluate_model(model, tokenizer, val_ds, val_contexts, evaluator, device)
        print(f"Validation Metrics: EM={val_metrics['EM']:.4f}, F1={val_metrics['F1-Score']:.4f}, BLEU={val_metrics['BLEU']:.4f}, ROUGE-L={val_metrics['ROUGE-L']:.4f}")
        
        if val_metrics['EM'] > best_em:
            best_em = val_metrics['EM']
            print(f"New best EM: {best_em:.4f}. Saving model...")
            model.save_pretrained("./best_centralized_text_model")
            
    print("Evaluating Test Set with Best Model...")
    if os.path.exists("./best_centralized_text_model"):
        # Reloading best adapter weights
        from peft import set_peft_model_state_dict
        model.load_adapter("./best_centralized_text_model", "default")
        
    test_metrics, _ = evaluate_model(model, tokenizer, test_ds, test_contexts, evaluator, device)
    print(f"Test Metrics: EM={test_metrics['EM']:.4f}, F1={test_metrics['F1-Score']:.4f}, BLEU={test_metrics['BLEU']:.4f}, ROUGE-L={test_metrics['ROUGE-L']:.4f}")

if __name__ == "__main__":
    run_centralized_training()
