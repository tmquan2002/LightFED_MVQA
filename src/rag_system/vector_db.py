import torch
import torch.nn as nn
import faiss
import numpy as np
import io
import os
import open_clip
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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

# Testing Rag System
if __name__ == "__main__":
    from datasets import load_dataset
    retriever = MedicalRetriever()
    
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad")['train'].select(range(50))
        print(f"[RAG Vector] Loaded {len(dataset)} samples")
        retriever.build_index_from_dataset(dataset)
        test_sample = dataset[-1]
        test_image = test_sample['image']
        
        print("\n--- RETRIEVAL ---")
        print(f"Original question: {test_sample['question']}")
        similar_cases = retriever.search_similar_cases(test_image, c=10)
        
        for i, case in enumerate(similar_cases):
            print(f"\nSimilar case #{i+1} (Vector Distance: {case['distance']:.4f}):")
            print(f"- Question: {case['question']}")
            print(f"- Old answer: {case['answer']}")
            
    except FileNotFoundError:
        print("Data not found")
