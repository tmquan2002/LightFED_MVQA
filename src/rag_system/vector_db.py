import torch
import faiss
import numpy as np
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class MedicalRetriever:
    """
    The Multimodal RAG system uses FAISS and CLIP models.
    Converting medical images into vectors and searching for similar cases.
    """
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RAG Vector] Init CLIP Encoder on {self.device.upper()}...")
        
        # Load CLIP model (Convert image to vector)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.vector_dim = self.model.config.projection_dim
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
        
    def _preprocess_image(self, image):
        """Process all image formats (such as the bytes dict error you just encountered) to the RGB standard."""
        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)
            
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image format")
            
        return image.convert("RGB")

    def get_image_embedding(self, image):
        """Convert an image into a 1D NumPy Array."""
        image = self._preprocess_image(image)
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
        Iterate through the entire dataset in batches, convert images into vectors, and store them in the FAISS repository.
        Batched CLIP encoding for significantly faster indexing.
        """
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
        print(f"\n[RAG Vector] Done, FAISS currently have {self.index.ntotal} vector.")

    def search_similar_cases(self, query_image, c=10):
        """
        Receive photos of new patients, find C cases with the most matching photos in the database.
        """
        if self.index.ntotal == 0:
            return []
            
        query_vector = self.get_image_embedding(query_image).astype('float32')
        distances, indices = self.index.search(query_vector, c)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                info = self.metadata[idx]
                info['distance'] = float(dist)
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