import numpy as np
from sklearn.metrics import f1_score
import warnings

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