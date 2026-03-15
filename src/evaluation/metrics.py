import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

class MedVQAEvaluator:
    """
    A specialized class for evaluating the Med-VQA model.
    Includes indicators for the Biogenesis (Open) and Classification (Closed) questions
    """
    def __init__(self):
        print("[Metric] Loading modules (BLEU, ROUGE)...")
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

    def _clean_text(self, text):
        """
        Text normalization: convert to lowercase, remove punctuation and extra spaces
        This is crucial for accurate Exact Match calculations
        """
        if not isinstance(text, str):
            return str(text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def evaluate_open_ended(self, predictions, references):
        """
        Calculate BLEU and ROUGE for Open-Ended (Description, Explanation) questions
        """
        if not predictions or not references:
            return {"BLEU": 0.0, "ROUGE-L": 0.0}

        formatted_refs = [[ref] for ref in references]
        
        bleu_results = self.bleu_metric.compute(predictions=predictions, references=formatted_refs)
        rouge_results = self.rouge_metric.compute(predictions=predictions, references=references)
        
        return {
            "BLEU": round(bleu_results.get("bleu", 0.0), 4),
            "ROUGE-L": round(rouge_results.get("rougeL", 0.0), 4)
        }

    def evaluate_closed_ended(self, predictions, references):
        """
        Calculate Exact Match, Accuracy, and F1-Score for closed-ended questions (Yes/No, Classification)
        """
        if not predictions or not references:
            return {"Exact Match": 0.0, "Accuracy": 0.0, "F1-Score": 0.0}

        cleaned_preds = [self._clean_text(p) for p in predictions]
        cleaned_refs = [self._clean_text(r) for r in references]
        
        # 1. Exact Match (EM)
        exact_matches = sum(1 for p, r in zip(cleaned_preds, cleaned_refs) if r in p or p in r)
        em_score = exact_matches / len(cleaned_preds)
        
        # 2. Accuracy & F1-Score
        acc = accuracy_score(cleaned_refs, cleaned_preds)
        
        # average='macro'
        _, _, f1, _ = precision_recall_fscore_support(
            cleaned_refs, cleaned_preds, average='macro', zero_division=0
        )
        
        return {
            "Exact Match": round(em_score, 4),
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4)
        }
    
# Test Metric
if __name__ == "__main__":
    evaluator = MedVQAEvaluator()
    
    ground_truths = ["yes", "mri", "right lung"]
    model_preds = ["Yes.", "mri scan", "right lung"]
    
    print("\nClosed-ended")
    closed_scores = evaluator.evaluate_closed_ended(model_preds[:2], ground_truths[:2])
    print(closed_scores)
    
    print("\nOpen-ended")
    open_scores = evaluator.evaluate_open_ended([model_preds[2]], [ground_truths[2]])
    print(open_scores)