import json
import os
# Fix duplicate OpenMP runtime library error on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import time
import torch
import random
import warnings
import io
import faiss
import numpy as np
import torch.nn as nn
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from sklearn.metrics import f1_score
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- Centralized Data Filtering ---
def is_closed_ended(answer) -> bool:
    """
    Determines if an answer is closed-ended (yes/no or short answers <= 2 words).
    """
    ans = str(answer).lower().strip()
    return ans in ['yes', 'no'] or len(ans.split()) <= 2

def filter_dataset(dataset: Dataset, question_type: str = "all", max_samples: int = None, seed: int = 42):
    """
    Filters the centralized dataset by question type and limits the number of samples.
    """
    filtered_dataset = dataset
    if question_type in ["closed", "open"]:
        indices = []
        for idx, item in enumerate(dataset):
            is_closed = is_closed_ended(item['answer'])
            if question_type == "closed" and is_closed:
                indices.append(idx)
            elif question_type == "open" and not is_closed:
                indices.append(idx)
        
        filtered_dataset = dataset.select(indices)
        print(f"[DataFilter] Filtered dataset to {question_type} questions: {len(dataset)} -> {len(filtered_dataset)} samples.")
        
    if max_samples is not None and len(filtered_dataset) > max_samples:
        import random
        random.seed(seed)
        indices = random.sample(range(len(filtered_dataset)), max_samples)
        # Sort indices to maintain order
        indices.sort()
        filtered_dataset = filtered_dataset.select(indices)
        print(f"[DataFilter] Limited dataset to {max_samples} samples.")
        
    return filtered_dataset

# --- src/evaluation/metrics.py ---
class MedVQAEvaluator:
    """
    A module for evaluating the performance of the Med-VQA model using standard metrics.
    """
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
class GatedFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super(GatedFusion, self).__init__()
        self.gate_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, image_embeds, text_embeds):
        combined = torch.cat((image_embeds, text_embeds), dim=1)
        gate = self.sigmoid(self.gate_layer(combined))
        gated_image = image_embeds * gate
        gated_text = text_embeds * (1 - gate)
        fused = self.fusion_layer(torch.cat((gated_image, gated_text), dim=1))
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused

class MedicalRetriever:
    def __init__(self, dataset_name=None, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.loaded_from_disk = False
        
        if dataset_name:
            save_dir = "./data/rag_index"
            index_path = os.path.join(save_dir, f"{dataset_name}_fusion.index")
            meta_path = os.path.join(save_dir, f"{dataset_name}_metadata.pt")
            fusion_path = os.path.join(save_dir, f"{dataset_name}_gated_fusion.pth")
            
            if os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(fusion_path):
                print(f"[RAG Vector] Loading pre-computed index & Gated Fusion for {dataset_name}...")
                import open_clip
                self.index = faiss.read_index(index_path)
                self.metadata = torch.load(meta_path, map_location='cpu')
                
                model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                self.model, _, self.preprocess_image_fn = open_clip.create_model_and_transforms(model_name)
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model.to(self.device).eval()
                
                self.fusion_model = GatedFusion(embed_dim=512).to(self.device)
                self.fusion_model.load_state_dict(torch.load(fusion_path, map_location=self.device))
                self.fusion_model.eval()
                
                self.loaded_from_disk = True
                print(f"[RAG Vector] Load complete. FAISS database contains {self.index.ntotal} cases.")
                return
                
        print(f"[RAG Vector] Init standard CLIP Encoder ({model_id}) on {self.device.upper()}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.vector_dim = self.model.config.projection_dim
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
        
    def _preprocess_image(self, image):
        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)
            
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image format")
            
        return image.convert("RGB")

    def get_image_embedding(self, image):
        image = self._preprocess_image(image)
        
        if self.loaded_from_disk:
            with torch.no_grad():
                proc_img = self.preprocess_image_fn(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(proc_img)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                if not isinstance(image_features, torch.Tensor):
                    if hasattr(image_features, 'pooler_output'):
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, 'image_embeds'):
                        image_features = image_features.image_embeds
                    else:
                        image_features = image_features.last_hidden_state[:, 0, :]
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    def build_index_from_dataset(self, dataset, batch_size=64):
        if self.loaded_from_disk:
            print(f"[RAG Vector] Skip dynamic build for {self.dataset_name} (loaded from disk).")
            return
            
        print(f"[RAG Vector] Loading {len(dataset)} samples into Vector Database (batch_size={batch_size})...")
        all_embeddings = []
        
        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, end_idx))
            
            images = [self._preprocess_image(img) for img in batch['image']]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, 'pooler_output'):
                        features = features.pooler_output
                    elif hasattr(features, 'image_embeds'):
                        features = features.image_embeds
                    else:
                        features = features.last_hidden_state[:, 0, :]
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(features.cpu().numpy())
            
            for j in range(end_idx - i):
                self.metadata.append({
                    "id": i + j,
                    "question": batch['question'][j],
                    "answer": batch['answer'][j]
                })
            
            if (i // batch_size) % 5 == 0:
                print(f"  Indexed {end_idx}/{len(dataset)} samples...", end="\r")
            
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        self.index.add(embeddings_matrix)
        print(f"\n[RAG Vector] Done, FAISS currently has {self.index.ntotal} vectors.")

    def search_similar_cases(self, query_image, query_question=None, c=10):
        if self.index.ntotal == 0:
            return []
            
        if self.loaded_from_disk and query_question is not None:
            query_image = self._preprocess_image(query_image)
            with torch.no_grad():
                proc_img = self.preprocess_image_fn(query_image).unsqueeze(0).to(self.device)
                tokens = self.tokenizer([str(query_question)]).to(self.device)
                
                img_feat = self.model.encode_image(proc_img)
                txt_feat = self.model.encode_text(tokens)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                
                query_vector = self.fusion_model(img_feat, txt_feat).cpu().numpy().astype('float32')
        else:
            query_vector = self.get_image_embedding(query_image).astype('float32')
            
        distances, indices = self.index.search(query_vector, c)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                info = dict(self.metadata[idx])
                info['distance'] = float(dist)
                
                if self.train_dataset is not None:
                    try:
                        info['image'] = self.train_dataset[info['id']]['image']
                    except Exception:
                        pass
                results.append(info)
                
        return results

# --- src/models/qwen_slm.py ---
class QwenMedVQA:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", use_4bit=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        quantization_config = None
        if use_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if use_4bit else self.device,
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _preprocess_image(self, image):
        if isinstance(image, dict) and 'bytes' in image:
            return Image.open(io.BytesIO(image['bytes'])).convert("RGB")
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def predict(self, image, question, retrieved_cases=None):
        target_img = self._preprocess_image(image)
        
        if retrieved_cases:
            content_block = []
            
            system_instruction = (
                "<System>\n"
                "You are an expert medical diagnostic assistant. Your task is to answer clinical questions "
                "based on an input pathology image and a set of retrieved similar cases.\n"
                "Strictly follow these rules:\n"
                "1. Analyze the provided [Image] and [Question].\n"
                "2. Evaluate the [Retrieved Evidence] from the knowledge base.\n"
                "Note that retrieved cases might have conflicting answers; use them as reference, "
                "but prioritize the visual features of the current image.\n"
                "3. Provide a concise and definitive answer.\n"
                "</System>\n\n"
                "<User>\n"
                "[Retrieved Evidence (Top-K)]\n"
                "Below are similar past cases from the clinical database:\n"
            )
            content_block.append({"type": "text", "text": system_instruction})
            
            def optimize_img(pil_img, max_res=336):
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)
                return pil_img
                
            messages = []
            for i, case in enumerate(retrieved_cases):
                try:
                    ref_img = self._preprocess_image(case['image'])
                    ref_img = optimize_img(ref_img)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ref_img},
                            {"type": "text", "text": f"Reference Case {i+1}:\nQuestion: {case['question']}"}
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
            
            target_img = optimize_img(target_img)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": target_img},
                    {"type": "text", "text": f"Current Question: {question}\nBased on the reference cases above, what is the correct answer for the current case? Answer concisely."}
                ]
            })
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": target_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=15 if retrieved_cases else 50,
                do_sample=False,        
                num_beams=1,            
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        raw_ans = output_text[0].strip()
        raw_ans = raw_ans.replace("</Assistant>", "").replace("<Assistant>", "").strip()
        return raw_ans

# --- Centralized Training Loop ---
def train_centralized(shared_slm, train_dataset, retriever, epochs, checkpoint_path):
    import torch.optim as optim
    from qwen_vl_utils import process_vision_info
    
    print(f"  Training centralized for {epochs} epoch(s)...")
    model = shared_slm.model
    processor = shared_slm.processor
    device = shared_slm.device
    
    model.train()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    indices = list(range(len(train_dataset)))
    
    def optimize_img(pil_img, max_res=336):
        from PIL import Image
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)
        return pil_img

    print(f"    Pre-computing RAG retrievals to avoid I/O bottleneck...")
    rag_cache = {}
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        retrieved = retriever.search_similar_cases(sample['image'], query_question=sample['question'], c=4)
        rag_cache[idx] = [case for case in retrieved if case['id'] != idx][:3]

    avg_loss = 0.0
    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()
        
        print(f"    Epoch {epoch+1}: Using {len(indices)} samples")
        for i, idx in enumerate(indices):
            sample = train_dataset[idx]
            question = sample['question']
            answer = str(sample['answer']).lower()
            
            similar_cases = rag_cache[idx]
            
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
            
            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs.loss / 16
                scaler.scale(loss).backward()
                
                accumulation_steps = 16
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(indices):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(**inputs)
                loss = outputs.loss / 16
                loss.backward()
                
                if (i + 1) % 16 == 0 or (i + 1) == len(indices):
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * 16
            steps += 1
            
            if i % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Step {i}/{len(indices)} | Loss: {loss.item():.4f}   ", end="\r")
                
        print()
        avg_loss = total_loss / steps if steps > 0 else 0.0
        print(f"  [PathVQA] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        lora_weights = {}
        for name, param in get_peft_model_state_dict(model).items():
            if getattr(param, "device", None) and param.device.type == 'meta':
                lora_weights[name] = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
            else:
                lora_weights[name] = param.clone().detach().cpu()
                
        torch.save({
            'epoch': epoch + 1,
            'weights': lora_weights,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"  [PathVQA] Auto-saved Epoch {epoch+1} checkpoint.")
        
    return avg_loss

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_scores_for_json(c_scores, o_scores, question_type="all"):
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
        
        is_closed = is_closed_ended(ground_truth)
        
        if question_type == "open" and is_closed:
            continue
        elif question_type == "closed" and not is_closed:
            continue
            
        similar_cases = retriever.search_similar_cases(image, query_question=question, c=3)
        pred = shared_slm.predict(image, question, retrieved_cases=similar_cases)
        
        if is_closed:
            closed_preds.append(pred); closed_refs.append(ground_truth)
        else:
            open_preds.append(pred); open_refs.append(ground_truth)
            
    infer_time = round(time.time() - start_infer_time, 2)
    print(f"\nInference Time: {infer_time} seconds")
    
    return evaluator.evaluate_closed_ended(closed_preds, closed_refs), evaluator.evaluate_open_ended(open_preds, open_refs), infer_time

def run_centralized_simulation(epochs, question_type="all", max_samples=None):
    print(f"\nSTARTING CENTRALIZED EXPERIMENTS (WITH RAG): {epochs} Epochs | Question Type = {question_type.upper()} | Max Samples = {max_samples if max_samples is not None else 'ALL'}")
    
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    
    path_vqa_train = filter_dataset(path_vqa["train"], question_type=question_type, max_samples=max_samples)
    path_vqa_test = path_vqa["test"]
    
    random.seed(int(time.time()))
    eval_seed = random.randint(0, 1000000)
    print(f"Using random evaluation seed: {eval_seed}")
    
    path_vqa_shuffled = path_vqa_test.shuffle(seed=eval_seed+1)
    
    eval_size = 50
    path_vqa_eval = path_vqa_shuffled.select(range(min(eval_size, len(path_vqa_test))))
    
    print(f"Evaluating with {len(path_vqa_eval)} PathVQA images (from official TEST set)")
    
    evaluator = MedVQAEvaluator()

    print("\n[RAG] Loading or Building Vector Databases from Training data only...")
    retriever_pv = MedicalRetriever("pathvqa_centralized")
    retriever_pv.train_dataset = path_vqa_train
    retriever_pv.build_index_from_dataset(path_vqa_train)
    
    os.makedirs("./data", exist_ok=True)
    file_name = f"eval_results_centralized_{epochs}epochs.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Epochs": epochs,
            "Question_Type": question_type.upper(),
            "Model_Type": "Centralized Learning with RAG"
        },
        "Training_Stats": {},
        "Results": {
            "Proposed (Centralized+RAG)": {}
        }
    }

    def save_current_progress(phase_name):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved progress '{phase_name}' to: {file_name}")

    print("\n>>> INITIALIZING SHARED QWEN2-VL ENGINE...")
    shared_slm = QwenMedVQA(use_4bit=True)
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

    print(f"\n==================== EXPERIMENT 1: VQA-RAD (TEMPORARILY SKIPPED) ====================")
    vr_t = 0.0
    results_dict["Training_Stats"]["VQA-RAD"] = "Skipped"
    results_dict["Results"]["Proposed (Centralized+RAG)"]["VQA-RAD"] = {
        "Accuracy": 0.0,
        "F1-Score": 0.0,
        "BLEU": 0.0,
        "ROUGE-L": 0.0
    }
    save_current_progress("VQA-RAD Experiment Skipped")

    print(f"\n==================== EXPERIMENT 2: PathVQA ====================")
    shared_slm.model.load_state_dict(initial_lora_weights, strict=False)
    
    os.makedirs("./model_checkpoints", exist_ok=True)
    checkpoint_pv = f"./model_checkpoints/lora_pv_centralized_{epochs}epochs.pt"

    start_train_time = time.time()
    final_loss = 0.0

    if os.path.exists(checkpoint_pv):
        print(f"\n[LOAD] Found existing trained weights checkpoint for PathVQA: {checkpoint_pv}")
        checkpoint = torch.load(checkpoint_pv, map_location='cpu')
        if isinstance(checkpoint, dict) and 'weights' in checkpoint:
            global_weights = checkpoint['weights']
            saved_epoch = checkpoint.get('epoch', epochs)
            print(f"Loading saved weights from epoch {saved_epoch} and skipping training...")
        else:
            global_weights = checkpoint
            print("Loading saved weights and skipping training...")
            
        shared_slm.model.load_state_dict(global_weights, strict=False)
    else:
        final_loss = train_centralized(shared_slm, path_vqa_train, retriever_pv, epochs, checkpoint_pv)

    total_train_time_pv = round(time.time() - start_train_time, 2)
    
    results_dict["Training_Stats"]["PathVQA"] = {
        "Final_Average_Loss": round(final_loss, 4) if isinstance(final_loss, (int, float)) else final_loss,
        "Training_Time_Seconds": total_train_time_pv
    }
    
    print("\nEvaluating PathVQA (Proposed Centralized+RAG)...")
    pv_c, pv_o, pv_t = evaluate_dataset(shared_slm, path_vqa_eval, evaluator, retriever_pv, question_type=question_type)
    
    results_dict["Results"]["Proposed (Centralized+RAG)"]["PathVQA"] = format_scores_for_json(pv_c, pv_o, question_type=question_type)
    results_dict["Results"]["Proposed (Centralized+RAG)"]["Inference_Time_Seconds"] = round(vr_t + pv_t, 2)
    save_current_progress("All Experiments Completed")
    print(f"\nCOMPLETED! Evaluation metrics saved to: {json_path}")

    clear_memory()

def get_user_setup():
    while True:
        try:
            epochs = int(input("\n1. Enter total Epochs for Centralized Training (e.g., 3, 5, 10): "))
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
    run_centralized_simulation(epochs_input, question_type=qtype_input, max_samples=max_samples_input)
