from src.models.qwen_slm import QwenMedVQA
from src.rag_system.vector_db import MedicalRetriever

class MedVQARAGPipeline:
    """
    Hệ thống RAG hoàn chỉnh: Kết hợp tìm kiếm vector (FAISS) và sinh văn bản (Qwen2-VL).
    """
    def __init__(self, use_4bit=True):
        print("[RAG-Pipeline] Initilize MED-VQA RAG Retriver")
        self.retriever = MedicalRetriever()
        self.slm = QwenMedVQA(use_4bit=use_4bit)

    def load_knowledge_base(self, dataset):
        """Loading hospital data into FAISS"""
        self.retriever.build_index_from_dataset(dataset)

    def predict(self, image, question, top_k=3):
        """
        Predict answer with RAG
        """
        # Find similar cases
        similar_cases = self.retriever.search_similar_cases(image, k=top_k)
        
        # Build prompt
        context_text = ""
        if similar_cases:
            context_text = "Here are some similar reference cases from the database:\n"
            for i, case in enumerate(similar_cases):
                context_text += f"- Reference {i+1}: Question: '{case['question']}' -> Answer: '{case['answer']}'\n"
        
        # Add context to actual question
        augmented_question = f"{context_text}\nNow, please answer this new Question: {question}"
        
        # In ra màn hình để bạn dễ theo dõi (khi debug)
        print("\n[RAG-Pipeline] RAG Prompt created, add to SLM")
        print(augmented_question)
        
        # Add new question and images to SLM
        answer = self.slm.predict(image=image, question=augmented_question)
        
        return answer

# Testing Pipeline
if __name__ == "__main__":
    from datasets import load_from_disk
    
    try:
        dataset = load_from_disk("./data/vqa_rad_subset_50")['train']
        pipeline = MedVQARAGPipeline(use_4bit=True)
        knowledge_base = dataset.select(range(49))
        pipeline.load_knowledge_base(knowledge_base)
        test_sample = dataset[49]
        prediction = pipeline.predict(
            image=test_sample['image'], 
            question=test_sample['question'], 
            top_k=2 # Lấy 2 ca giống nhất
        )
        
        print(f"Original question: {test_sample['question']}")
        print(f"Ground Truth: {test_sample['answer']}")
        print(f"SLM (+ RAG) prediction: {prediction}")
        
    except FileNotFoundError:
        print("Data not found, please run file data_subset_generator.py first")