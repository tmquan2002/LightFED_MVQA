import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import faiss
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
import io
import os
import random
import numpy as np

# Ensure PIL loads truncated images (common in PathVQA)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Directory to save checkpoints and FAISS indices
SAVE_DIR = "./data/rag_index"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
        # L2 Normalize the fused vector
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused

def train_fusion_model(dataset_name, train_dataset, epochs=3, batch_size=64, device="cuda"):
    print(f"\n--- Training Gated Fusion Model for {dataset_name.upper()} ---")
    
    # Load BiomedCLIP model
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    biomed_clip, _, preprocess_image = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    biomed_clip.to(device).eval()
    
    # Initialize fusion network
    fusion_model = GatedFusion(embed_dim=512).to(device)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=2e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    def collate_fn(batch):
        images_proc = []
        questions, answers = [], []
        for item in batch:
            try:
                # Convert PIL image to RGB
                img = item['image']
                if isinstance(img, dict) and 'bytes' in img:
                    img = Image.open(io.BytesIO(img['bytes']))
                img = img.convert("RGB") if img.mode != "RGB" else img
                
                images_proc.append(preprocess_image(img))
                questions.append(str(item['question']))
                answers.append(str(item['answer']))
            except Exception:
                continue
        if not images_proc:
            return None
        return torch.stack(images_proc), tokenizer(questions), tokenizer(answers)
    
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    temperature = 0.07
    for epoch in range(epochs):
        fusion_model.train()
        total_loss = 0.0
        batches_processed = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
                
            images, questions, answers = batch
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Extract static representations from BiomedCLIP
                img_feats = biomed_clip.encode_image(images)
                quest_feats = biomed_clip.encode_text(questions)
                ans_feats = biomed_clip.encode_text(answers)
                
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                quest_feats /= quest_feats.norm(dim=-1, keepdim=True)
                ans_feats /= ans_feats.norm(dim=-1, keepdim=True)
            
            # Fuse image + question, and compute contrastive loss against answer
            fused_vec = fusion_model(img_feats, quest_feats)
            logits = (fused_vec @ ans_feats.T) / temperature
            labels = torch.arange(len(images)).to(device)
            
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches_processed += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] | Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
        print(f"  -> Finished Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
    # Save weights
    weight_path = os.path.join(SAVE_DIR, f"{dataset_name}_gated_fusion.pth")
    torch.save(fusion_model.state_dict(), weight_path)
    print(f"Saved Gated Fusion weights to: {weight_path}")
    return fusion_model

def build_vector_database(dataset_name, dataset, fusion_model, device="cuda"):
    print(f"\n--- Building FAISS Database for {dataset_name.upper()} ---")
    
    # Initialize BiomedCLIP for database encoding
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    biomed_clip, _, preprocess_image = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    biomed_clip.to(device).eval()
    fusion_model.eval()
    
    faiss_index = faiss.IndexFlatL2(512)
    metadata = []
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            try:
                img = item['image']
                if isinstance(img, dict) and 'bytes' in img:
                    img = Image.open(io.BytesIO(img['bytes']))
                img = img.convert("RGB") if img.mode != "RGB" else img
                
                # Preprocess image and question
                proc_img = preprocess_image(img).unsqueeze(0).to(device)
                tokens = tokenizer([str(item['question'])]).to(device)
                
                img_feat = biomed_clip.encode_image(proc_img)
                txt_feat = biomed_clip.encode_text(tokens)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                
                # Fuse and add to FAISS
                fused_vec = fusion_model(img_feat, txt_feat).cpu().numpy().astype('float32')
                faiss_index.add(fused_vec)
                
                metadata.append({
                    "id": i,
                    "question": str(item['question']),
                    "answer": str(item['answer'])
                })
            except Exception:
                continue
                
            if (i + 1) % 2000 == 0 or (i + 1) == len(dataset):
                print(f"  Indexed {i+1}/{len(dataset)} samples...")
                
    # Save index and metadata
    index_path = os.path.join(SAVE_DIR, f"{dataset_name}_fusion.index")
    meta_path = os.path.join(SAVE_DIR, f"{dataset_name}_metadata.pt")
    
    faiss.write_index(faiss_index, index_path)
    torch.save(metadata, meta_path)
    
    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved Metadata to: {meta_path}")
    print(f"Total entries indexed: {faiss_index.ntotal}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 1. Load datasets from Hugging Face
    print("Loading PathVQA and VQA-RAD datasets...")
    path_vqa = load_dataset("flaviagiammarino/path-vqa")
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    
    # 2. Train PathVQA Fusion & Build database
    path_fusion_model = train_fusion_model("pathvqa", path_vqa["train"], epochs=3, batch_size=64, device=device)
    build_vector_database("pathvqa", path_vqa["train"], path_fusion_model, device=device)
    
    # 3. Train VQA-RAD Fusion & Build database
    vqa_fusion_model = train_fusion_model("vqarad", vqa_rad["train"], epochs=3, batch_size=64, device=device)
    build_vector_database("vqarad", vqa_rad["train"], vqa_fusion_model, device=device)
    
    print("\n🎉 OFFLINE RAG TRAINING & INDEXING COMPLETED SUCCESSFULLY!")
