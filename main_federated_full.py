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
import open_clip
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- src/data_processing/data_splitter.py ---
class FederatedDataSplitter:
    """
    A utility class designed to partition a centralized dataset into multiple 
    smaller subsets for simulated Federated Learning (FL) clients.
    """
    @staticmethod
    def is_closed_ended(answer) -> bool:
        """
        Determines if an answer is closed-ended (yes/no or short answers <= 2 words).
        """
        ans = str(answer).lower().strip()
        return ans in ['yes', 'no'] or len(ans.split()) <= 2

    def __init__(self, dataset: Dataset, num_clients: int = 2, seed: int = None, question_type: str = "all", max_samples: int = None):
        self.num_clients = num_clients
        self.seed = seed
        
        # 1. Filter by question type if requested
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
            
        # 2. Limit the number of samples if requested
        if max_samples is not None and len(filtered_dataset) > max_samples:
            import random
            if seed is not None:
                random.seed(seed)
            else:
                random.seed(42)  # default seed for stable subsetting
            
            indices = random.sample(range(len(filtered_dataset)), max_samples)
            # Sort indices to maintain order
            indices.sort()
            filtered_dataset = filtered_dataset.select(indices)
            print(f"[DataSplitter] Limited dataset to {max_samples} samples.")
            
        self.dataset = filtered_dataset
        
    def split_iid(self):
        """
        Performs Independent and Identically Distributed (IID) data splitting.
        """
        import random
        if self.seed is None:
            shuffle_seed = random.randint(0, 1000000)
        else:
            shuffle_seed = self.seed
        
        shuffled_dataset = self.dataset.shuffle(seed=shuffle_seed)
        split_size = len(shuffled_dataset) // self.num_clients
        print(f"[DataSplitter] IID split with seed {shuffle_seed}")
        
        client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = len(shuffled_dataset) if i == self.num_clients - 1 else (i + 1) * split_size
            subset = shuffled_dataset.select(range(start_idx, end_idx))
            client_datasets.append(subset)
            
        return client_datasets

    def split_non_iid(self, alpha: float = 0.5):
        """
        Performs Non-IID data splitting 
        using Dirichlet distribution.
        """
        import random
        if self.seed is None:
            dirichlet_seed = random.randint(0, 1000000)
        else:
            dirichlet_seed = self.seed
            
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
        for indices in client_indices:
            np.random.shuffle(indices)
            if len(indices) == 0:
                indices = [np.random.randint(0, len(self.dataset))]
                
            client_datasets.append(self.dataset.select(indices))
            
        print(f"[DataSplitter] Non-IID splitted (alpha={alpha}).")
        for i, ds in enumerate(client_datasets):
            print(f"Hospital {i+1} receives: {len(ds)} photos.")
            
        return client_datasets

    def split_by_question_type(self):
        """
        Partitions the dataset such that a subset of clients receive only closed-ended
        questions, and the remaining clients receive only open-ended questions.
        """
        import random
        if self.seed is not None:
            random.seed(self.seed)
            
        closed_indices = []
        open_indices = []
        
        for idx, item in enumerate(self.dataset):
            if self.is_closed_ended(item['answer']):
                closed_indices.append(idx)
            else:
                open_indices.append(idx)
                
        # Split clients into two groups: first half closed-ended, second half open-ended
        num_closed_clients = self.num_clients // 2
        num_open_clients = self.num_clients - num_closed_clients
        
        print(f"[DataSplitter] Splitting by question type across clients:")
        print(f"  - {num_closed_clients} clients get closed-ended ({len(closed_indices)} total samples)")
        print(f"  - {num_open_clients} clients get open-ended ({len(open_indices)} total samples)")
        
        # Shuffle indices
        random.shuffle(closed_indices)
        random.shuffle(open_indices)
        
        client_datasets = [None] * self.num_clients
        
        # Distribute closed-ended
        if num_closed_clients > 0:
            closed_split_size = len(closed_indices) // num_closed_clients
            for i in range(num_closed_clients):
                start_idx = i * closed_split_size
                end_idx = len(closed_indices) if i == num_closed_clients - 1 else (i + 1) * closed_split_size
                client_datasets[i] = self.dataset.select(closed_indices[start_idx:end_idx])
                
        # Distribute open-ended
        if num_open_clients > 0:
            open_split_size = len(open_indices) // num_open_clients
            for i in range(num_open_clients):
                client_idx = num_closed_clients + i
                start_idx = i * open_split_size
                end_idx = len(open_indices) if i == num_open_clients - 1 else (i + 1) * open_split_size
                client_datasets[client_idx] = self.dataset.select(open_indices[start_idx:end_idx])
                
        for i, ds in enumerate(client_datasets):
            print(f"Hospital {i+1} receives: {len(ds)} photos.")
            
        return client_datasets
# --- src/federated/server.py ---
class FederatedServer:
    """
    The Central Server in Federated Learning.
    Collect LoRA weights from clients and aggregate them using FedAvg.
    """
    def __init__(self):
        self.global_weights = None
        print("\n[Server] The central server started up")

    def aggregate_weights(self, client_weights_list, client_sizes=None):
        """
        Federated Averaging Algorithm
        Calculate the average of the weight matrices from the clients.
        Supports optional dataset-weighted aggregation.
        """
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
            # Weighted average aggregation
            weighted_tensors = [t.to(torch.float32) * w for t, w in zip(tensors, weights)]
            avg_tensor = torch.stack(weighted_tensors).sum(dim=0)
            averaged_weights[key] = avg_tensor.to(tensors[0].dtype)
            
        self.global_weights = averaged_weights
        print("[Server] Done, new Global Model created")
        
        return self.global_weights
# --- src/evaluation/metrics.py ---
class MedVQAEvaluator:
    """
    A module for evaluating the performance of the Med-VQA model using standard metrics.
    """
    def __init__(self):
        pass

    def evaluate_closed_ended(self, preds, refs):
        """
        Evaluate closed-ended questions (Yes/No or 1-2 word answers).
        Use Substring Match to overcome the "Generative Mismatch" issue of LLM/SLM.
        """
        cleaned_preds = [str(p).lower().strip() for p in preds]
        cleaned_refs = [str(r).lower().strip() for r in refs]

        correct = 0
        mapped_preds = []

        for p, r in zip(cleaned_preds, cleaned_refs):
            # If the ground truth answer is contained within the AI's response -> Count as CORRECT
            # Example: r = "yes", p = "yes, there is a tumor." -> CORRECT
            if r in p or p in r:
                correct += 1
                mapped_preds.append(r) # Normalize for F1-Score calculation
            else:
                mapped_preds.append(p)

        # Calculate Accuracy
        accuracy = correct / len(refs) if refs else 0.0

        # Calculate Macro F1-Score (Ignore warnings about division by zero if a class doesn't appear)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(cleaned_refs, mapped_preds, average='macro', zero_division=0)

        return {"Accuracy": accuracy, "F1-Score": f1}

    def evaluate_open_ended(self, preds, refs):
        """
        Evaluate open-ended questions (Description, Explanation).
        Measure vocabulary similarity using BLEU and ROUGE-L.
        """
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

            # Estimate basic vocabulary overlap (Intersection)
            common_tokens = set(p_tokens).intersection(set(r_tokens))
            precision = len(common_tokens) / len(p_tokens)
            recall = len(common_tokens) / len(r_tokens)

            if precision + recall == 0:
                bleu, rouge = 0.0, 0.0
            else:
                # Calculate BLEU (Focus on Precision + Penalty if AI response is too short)
                brevity_penalty = 1.0 if len(p_tokens) > len(r_tokens) else np.exp(1 - len(r_tokens) / len(p_tokens))
                bleu = brevity_penalty * precision
                
                # Calculate ROUGE-L (Focus on Recall)
                rouge = recall

            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge)

        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge_l = np.mean(rouge_l_scores) if rouge_l_scores else 0.0

        return {"BLEU": avg_bleu, "ROUGE-L": avg_rouge_l}

    def print_results_table(self, method_name, vqa_rad_closed, vqa_rad_open, path_vqa_closed, path_vqa_open):
        "Support printing result rows to Terminal (if needed)"
        vr_acc = vqa_rad_closed.get('Accuracy', 0) * 100
        vr_f1 = vqa_rad_closed.get('F1-Score', 0) * 100
        vr_bleu = vqa_rad_open.get('BLEU', 0) * 100
        vr_rouge = vqa_rad_open.get('ROUGE-L', 0) * 100

        pv_acc = path_vqa_closed.get('Accuracy', 0) * 100
        pv_f1 = path_vqa_closed.get('F1-Score', 0) * 100
        pv_bleu = path_vqa_open.get('BLEU', 0) * 100
        pv_rouge = path_vqa_open.get('ROUGE-L', 0) * 100

        print(f"| {method_name:<20} | {vr_acc:<6.1f} {vr_f1:<5.1f} {vr_bleu:<6.1f} {vr_rouge:<5.1f} | {pv_acc:<6.1f} {pv_f1:<5.1f} {pv_bleu:<6.1f} {pv_rouge:<5.1f} |")
# --- src/rag_system/vector_db.py ---
class GatedFusion(nn.Module):
    """
    Fuses image and text embeddings from BiomedCLIP using a gating mechanism
    to produce a combined multimodal representation for querying the vector DB.
    """
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
    """
    The Multimodal RAG system uses FAISS and either BiomedCLIP (with Gated Fusion)
    or standard CLIP models. Supports loading index files from disk.
    """
    def __init__(self, dataset_name=None, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.loaded_from_disk = False
        
        # Check if pre-computed index exists on disk
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
                
                # Load BiomedCLIP
                model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                self.model, _, self.preprocess_image_fn = open_clip.create_model_and_transforms(model_name)
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model.to(self.device).eval()
                
                # Load Fusion model
                self.fusion_model = GatedFusion(embed_dim=512).to(self.device)
                self.fusion_model.load_state_dict(torch.load(fusion_path, map_location=self.device))
                self.fusion_model.eval()
                
                self.loaded_from_disk = True
                print(f"[RAG Vector] Load complete. FAISS database contains {self.index.ntotal} cases.")
                return
                
        # Fallback to standard OpenAI CLIP if index not found or dataset_name is None
        print(f"[RAG Vector] Init standard CLIP Encoder ({model_id}) on {self.device.upper()}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.vector_dim = self.model.config.projection_dim
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
        
    def _preprocess_image(self, image):
        """Process all image formats to RGB standard."""
        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)
            
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image format")
            
        return image.convert("RGB")

    def get_image_embedding(self, image):
        """Convert an image into a 1D NumPy Array using the appropriate CLIP model."""
        image = self._preprocess_image(image)
        
        if self.loaded_from_disk:
            # BiomedCLIP image encoding
            with torch.no_grad():
                proc_img = self.preprocess_image_fn(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(proc_img)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        else:
            # Standard CLIP image encoding
            # pyrefly: ignore [unexpected-keyword]
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
        """
        Build index dynamically from dataset. Skip if already loaded from disk.
        """
        if self.loaded_from_disk:
            print(f"[RAG Vector] Skip dynamic build for {self.dataset_name} (loaded from disk).")
            return
            
        print(f"[RAG Vector] Loading {len(dataset)} samples into Vector Database (batch_size={batch_size})...")
        all_embeddings = []
        
        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, end_idx))
            
            images = [self._preprocess_image(img) for img in batch['image']]
            # pyrefly: ignore [unexpected-keyword]
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
        """
        Find C cases matching the visual and text queries.
        """
        if self.index.ntotal == 0:
            return []
            
        if self.loaded_from_disk and query_question is not None:
            # Gated Fusion multimodal retrieval
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
            # Fallback to image-only retrieval
            query_vector = self.get_image_embedding(query_image).astype('float32')
            
        distances, indices = self.index.search(query_vector, c)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                info = dict(self.metadata[idx])
                info['distance'] = float(dist)
                
                # Resolve image from training dataset if available
                if self.train_dataset is not None:
                    try:
                        info['image'] = self.train_dataset[info['id']]['image']
                    except Exception:
                        pass
                results.append(info)
                
        return results

# --- src/models/qwen_slm.py ---
class QwenMedVQA:
    """
    A wrapper class for the Qwen2-VL Small Language Model (SLM) tailored for 
    Medical Visual Question Answering (Med-VQA). It handles model initialization 
    (with optional 4-bit quantization for memory efficiency), image preprocessing, 
    and text generation while preventing memory leaks during inference.
    """
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", use_4bit=True):
        """
        Initializes the Qwen2-VL model and its corresponding processor.
        """
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
        """
        Standardizes the input image format to an RGB PIL Image.
        Handles raw bytes dictionaries (from Hugging Face datasets) and file paths.
        """
        if isinstance(image, dict) and 'bytes' in image:
            return Image.open(io.BytesIO(image['bytes'])).convert("RGB")
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def predict(self, image, question, retrieved_cases=None):
        """
        Generates an answer for a given medical image and question.
        Supports passing retrieved visual RAG cases for prompt augmentation.
        """
        from PIL import Image
        
        target_img = self._preprocess_image(image)
        
        if retrieved_cases:
            # Multi-image Visual RAG prompting (identical to testPathVQA.ipynb structure)
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
            
            # Helper to resize images to avoid VRAM overload
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
            # Fallback to direct direct VQA prompt
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

        # Prevents memory leak during inference loop   
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
# --- main_federated.py ---
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

        # Pre-compute RAG cases to avoid I/O bottleneck
        print(f"    [{self.client_id}] Pre-computing RAG retrievals to avoid I/O bottleneck...")
        rag_cache = {}
        for idx in range(len(self.local_dataset)):
            sample = self.local_dataset[idx]
            retrieved = self.retriever.search_similar_cases(sample['image'], query_question=sample['question'], c=4)
            rag_cache[idx] = [case for case in retrieved if case['id'] != idx][:3]

        for epoch in range(epochs):
            random.shuffle(indices)
            epoch_indices = indices  # Cap training samples per round removed
            print(f"    [{self.client_id}] Using {len(epoch_indices)}/{len(indices)} samples this round")
            for i, idx in enumerate(epoch_indices):
                sample = self.local_dataset[idx]
                question = sample['question']
                answer = str(sample['answer']).lower()
                
                # Use pre-computed retrieved cases
                similar_cases = rag_cache[idx]
                
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
                        loss = outputs.loss / 16  # Gradient accumulation scale (steps=16)
                    scaler.scale(loss).backward()
                    
                    accumulation_steps = 16
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_indices):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss / 16
                    loss.backward()
                    
                    if (i + 1) % 16 == 0 or (i + 1) == len(epoch_indices):
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * 16  # Restore original loss scale for stats
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
        
        is_closed = ground_truth in ['yes', 'no'] or len(ground_truth.split()) <= 2
        
        if question_type == "open" and is_closed:
            continue
        elif question_type == "closed" and not is_closed:
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
    file_name = f"eval_results_{num_clients}clients_{num_rounds}rounds_{split_type.upper()}_a{alpha_str}.json"
    json_path = os.path.join("./data", file_name)
    
    results_dict = {
        "Experiment_Config": {
            "Num_Clients": num_clients,
            "Max_Rounds_Configured": num_rounds,
            "Local_Epochs": epochs,
            "Split_Type": split_type.upper(),
            "Alpha": alpha if split_type == 'non-iid' else "NA",
            "Model_Type": "Federated Learning with RAG"
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
    
    checkpoint_pv = f"./model_checkpoints/lora_pv_{num_clients}clients_{num_rounds}rounds_{epochs}epochs_{split_type}_a{alpha_str}.pt"

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