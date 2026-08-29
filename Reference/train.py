#!/usr/bin/env python3
"""
train.py

Generative RAG Medical VQA training script for PathVQA.
Uses:
- Frozen BiomedCLIP for image-question embedding.
- FAISS on CPU for retrieval of relevant cases from the training set.
- Qwen2.5-0.5B-Instruct + LoRA for answer generation.
- Evaluation metrics: Exact Match (EM), F1 Score, BLEU, ROUGE-L.

Optimized to run under 7GB VRAM by precomputing embeddings, releasing BiomedCLIP,
and applying LoRA on Qwen.
"""

import os
import gc
import argparse
import random
import torch
from torch.utils.data import Dataset, DataLoader
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import open_clip
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Set up logging/printing helper
def log(msg):
    print(f"[RAG-VQA] {msg}")

# Seed setting
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Precompute embeddings
def precompute_embeddings(csv_path, data_dir, model, preprocess, tokenizer, device):
    df = pd.read_csv(csv_path)
    embeddings = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {csv_path.name}"):
        img_path = data_dir / row['image_path']
        question = str(row['question'])
        
        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            log(f"Warning: failed to load image {img_path}: {e}. Using zero tensor.")
            image_input = torch.zeros(1, 3, 224, 224).to(device)
            
        # Tokenize question
        text_input = tokenizer([question]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # L2-normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Combine by concatenation
            combined = torch.cat([image_features, text_features], dim=-1)
            embeddings.append(combined.cpu().numpy()[0])
            
    return np.array(embeddings)

# Format retrieved documents
def format_retrieved_knowledge(df, indices):
    docs = []
    for rank, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        docs.append(f"- Question: {row['question']}\n  Answer: {row['answer']}")
    return "\n".join(docs)

# Custom PyTorch Dataset
class MedVQADataset(Dataset):
    def __init__(self, df, retrieved_contexts, tokenizer, max_length=512):
        self.df = df
        self.retrieved_contexts = retrieved_contexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        context = self.retrieved_contexts[idx]
        answer = row['answer']
        
        prompt = f"Question:\n{question}\n\nRetrieved Knowledge:\n{context}\n\nAnswer:\n"
        
        # Tokenize prompt and target answer
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(str(answer), add_special_tokens=False) + [self.tokenizer.eos_token_id]
        
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # Truncate if exceeds max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

# Collate function for dynamic padding
def get_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        max_len = max(len(x) for x in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
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

# Evaluation metric helpers
def compute_exact_match(prediction, ground_truth):
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

def compute_f1(prediction, ground_truth):
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

def compute_bleu(prediction, ground_truth):
    ref = [ground_truth.strip().lower().split()]
    hyp = prediction.strip().lower().split()
    smooth = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smooth)

def compute_rouge_l(prediction, ground_truth, scorer):
    scores = scorer.score(ground_truth.strip().lower(), prediction.strip().lower())
    return scores['rougeL'].fmeasure

# Evaluation loop
def evaluate_model(model, tokenizer, test_df, test_contexts, scorer, device, desc="Evaluating"):
    model.eval()
    predictions = []
    
    log(f"Running generation on {desc} set...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question = row['question']
        context = test_contexts[idx]
        
        prompt = f"Question:\n{question}\n\nRetrieved Knowledge:\n{context}\n\nAnswer:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0
            )
            
        # Decode only the generated part
        prompt_len = inputs['input_ids'].shape[1]
        gen_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(gen_text)
        
    # Calculate metrics
    ems = []
    f1s = []
    bleus = []
    rouges = []
    
    for pred, row in zip(predictions, test_df.itertuples()):
        gt = row.answer
        ems.append(compute_exact_match(pred, gt))
        f1s.append(compute_f1(pred, gt))
        bleus.append(compute_bleu(pred, gt))
        rouges.append(compute_rouge_l(pred, gt, scorer))
        
    metrics = {
        "EM": np.mean(ems),
        "F1": np.mean(f1s),
        "BLEU": np.mean(bleus),
        "ROUGE-L": np.mean(rouges)
    }
    
    return metrics, predictions

def main():
    parser = argparse.ArgumentParser(description="Train Generative RAG Med-VQA on PathVQA")
    parser.add_argument("--data_dir", type=str, default="processed_data/PathVQA/centralized",
                        help="Path to centralized PathVQA dataset folder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Generator LLM model path/name on HF")
    parser.add_argument("--biomed_model_name", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        help="BiomedCLIP model checkpoint")
    parser.add_argument("--output_dir", type=str, default="code/PathVQA/results",
                        help="Directory to save the checkpoints and predictions")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of retrieved cases to include as knowledge context")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "validation.csv"
    test_csv = data_dir / "test.csv"
    
    assert train_csv.exists(), f"Train CSV not found at {train_csv}"
    assert val_csv.exists(), f"Validation CSV not found at {val_csv}"
    assert test_csv.exists(), f"Test CSV not found at {test_csv}"
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    # ----------------------------------------------------
    # Step 1: Precompute Embeddings using BiomedCLIP
    # ----------------------------------------------------
    train_cache = output_dir / "train_biomed_embeddings.npy"
    val_cache = output_dir / "val_biomed_embeddings.npy"
    test_cache = output_dir / "test_biomed_embeddings.npy"
    
    if train_cache.exists() and val_cache.exists() and test_cache.exists():
        log("Loading precomputed BiomedCLIP embeddings from cache...")
        train_features = np.load(train_cache)
        val_features = np.load(val_cache)
        test_features = np.load(test_cache)
    else:
        log("Loading BiomedCLIP model to compute embeddings...")
        biomed_model, preprocess = open_clip.create_model_from_pretrained(args.biomed_model_name)
        biomed_tokenizer = open_clip.get_tokenizer(args.biomed_model_name)
        biomed_model.to(device)
        biomed_model.eval()
        
        log("Computing train embeddings...")
        train_features = precompute_embeddings(train_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(train_cache, train_features)
        
        log("Computing validation embeddings...")
        val_features = precompute_embeddings(val_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(val_cache, val_features)
        
        log("Computing test embeddings...")
        test_features = precompute_embeddings(test_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(test_cache, test_features)
        
        # Free GPU memory completely
        del biomed_model
        torch.cuda.empty_cache()
        gc.collect()
        log("Cleared BiomedCLIP from memory.")
        
    # ----------------------------------------------------
    # Step 2: Build FAISS Index and Perform Retrieval
    # ----------------------------------------------------
    log("Building FAISS index on train features...")
    # Using L2-normalized vectors and Inner Product for Cosine Similarity search
    def l2_norm_rows(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1, norms)
        
    train_features = l2_norm_rows(train_features)
    val_features = l2_norm_rows(val_features)
    test_features = l2_norm_rows(test_features)
    
    index = faiss.IndexFlatIP(1024)  # 1024 dimensions
    index.add(train_features)
    
    log(f"Performing RAG retrieval (k={args.k})...")
    # For training data, we query the index but exclude the query sample itself to prevent ground-truth leakage.
    train_contexts = []
    # Search for k+1 neighbors in training set
    D_train, I_train = index.search(train_features, args.k + 1)
    for idx in range(len(train_df)):
        retrieved_indices = []
        for match_idx in I_train[idx]:
            if match_idx == idx:
                continue
            retrieved_indices.append(match_idx)
            if len(retrieved_indices) == args.k:
                break
        # Fallback if somehow len < k
        if len(retrieved_indices) < args.k:
            for match_idx in I_train[idx]:
                if match_idx not in retrieved_indices:
                    retrieved_indices.append(match_idx)
                if len(retrieved_indices) == args.k:
                    break
        train_contexts.append(format_retrieved_knowledge(train_df, retrieved_indices))
        
    # For validation data, we retrieve top-k closest from training set
    val_contexts = []
    D_val, I_val = index.search(val_features, args.k)
    for idx in range(len(val_df)):
        val_contexts.append(format_retrieved_knowledge(train_df, I_val[idx]))
        
    # For test data, we just retrieve top-k closest from training set
    test_contexts = []
    D_test, I_test = index.search(test_features, args.k)
    for idx in range(len(test_df)):
        test_contexts.append(format_retrieved_knowledge(train_df, I_test[idx]))
        
    log("RAG retrieval contexts prepared successfully.")
    
    # ----------------------------------------------------
    # Step 3: Load Qwen Generator LLM & Apply LoRA
    # ----------------------------------------------------
    log(f"Loading tokenizer & LLM: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set model dtype to float16 / bfloat16 to fit under VRAM limit
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # PEFT LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # ----------------------------------------------------
    # Step 4: Setup Dataset & DataLoader
    # ----------------------------------------------------
    train_dataset = MedVQADataset(train_df, train_contexts, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(tokenizer)
    )
    
    # Scorer for Rouge
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # ----------------------------------------------------
    # Step 5: Training Loop
    # ----------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    log("Starting training...")
    best_em = -1.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        log(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")
        
        # Evaluate model after each epoch on validation set
        val_metrics, val_predictions = evaluate_model(model, tokenizer, val_df, val_contexts, scorer, device, desc="Validation")
        log(f"Epoch {epoch} Validation Metrics: EM={val_metrics['EM']:.4f}, F1={val_metrics['F1']:.4f}, BLEU={val_metrics['BLEU']:.4f}, ROUGE-L={val_metrics['ROUGE-L']:.4f}")
        
        # Save best model based on Validation EM
        if val_metrics['EM'] > best_em:
            best_em = val_metrics['EM']
            log(f"New best Validation Exact Match: {best_em:.4f}. Saving adapter weights...")
            model.save_pretrained(output_dir / "best_lora_adapter")
            
            # Save validation predictions just in case
            val_results_df = val_df.copy()
            val_results_df['retrieved_knowledge'] = val_contexts
            val_results_df['prediction'] = val_predictions
            val_results_df['EM'] = [compute_exact_match(p, r.answer) for p, r in zip(val_predictions, val_df.itertuples())]
            val_results_df['F1'] = [compute_f1(p, r.answer) for p, r in zip(val_predictions, val_df.itertuples())]
            val_results_df['BLEU'] = [compute_bleu(p, r.answer) for p, r in zip(val_predictions, val_df.itertuples())]
            val_results_df['ROUGE-L'] = [compute_rouge_l(p, r.answer, scorer) for p, r in zip(val_predictions, val_df.itertuples())]
            val_results_df.to_csv(output_dir / "best_val_predictions.csv", index=False)
            
            # Also run evaluation on Test set and save best predictions
            test_metrics, test_predictions = evaluate_model(model, tokenizer, test_df, test_contexts, scorer, device, desc="Test")
            log(f"Epoch {epoch} Test Metrics: EM={test_metrics['EM']:.4f}, F1={test_metrics['F1']:.4f}, BLEU={test_metrics['BLEU']:.4f}, ROUGE-L={test_metrics['ROUGE-L']:.4f}")
            
            test_results_df = test_df.copy()
            test_results_df['retrieved_knowledge'] = test_contexts
            test_results_df['prediction'] = test_predictions
            test_results_df['EM'] = [compute_exact_match(p, r.answer) for p, r in zip(test_predictions, test_df.itertuples())]
            test_results_df['F1'] = [compute_f1(p, r.answer) for p, r in zip(test_predictions, test_df.itertuples())]
            test_results_df['BLEU'] = [compute_bleu(p, r.answer) for p, r in zip(test_predictions, test_df.itertuples())]
            test_results_df['ROUGE-L'] = [compute_rouge_l(p, r.answer, scorer) for p, r in zip(test_predictions, test_df.itertuples())]
            test_results_df.to_csv(output_dir / "best_predictions.csv", index=False)
            log(f"Saved test predictions and scores to {output_dir / 'best_predictions.csv'}")

    log("Training completed successfully!")

if __name__ == "__main__":
    main()
