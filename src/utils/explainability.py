import numpy as np
from typing import List, Dict, Any, Tuple, Callable

class TextFeatureAttribution:
    """
    Computes token-level feature attribution scores for text classification.
    Uses a perturbation-based Leave-One-Out (LOO) approach to approximate Shapley values.
    """
    def __init__(self, classify_fn: Callable[[str, List[str]], Dict[str, Any]]):
        """
        Args:
            classify_fn: Callable that takes (text, candidate_labels) and returns a dict with predictions
        """
        self.classify_fn = classify_fn

    def explain(
        self,
        text: str,
        target_label: str,
        candidate_labels: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Calculates attribution score for each word in text for the target class.
        
        Returns:
            List of (word, score) pairs
        """
        words = text.split()
        if len(words) == 0:
            return []
            
        # Get baseline prediction score for the full text
        baseline_res = self.classify_fn(text, candidate_labels)
        baseline_prob = 0.0
        for pred in baseline_res["predictions"]:
            if pred["label"] == target_label:
                baseline_prob = pred["probability"]
                break
                
        attributions = []
        
        # Perturb by removing each word one-by-one
        for i in range(len(words)):
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = " ".join(perturbed_words)
            
            if not perturbed_text.strip():
                # If text is empty, contribution is baseline difference from zero
                perturbed_prob = 0.0
            else:
                p_res = self.classify_fn(perturbed_text, candidate_labels)
                perturbed_prob = 0.0
                for pred in p_res["predictions"]:
                    if pred["label"] == target_label:
                        perturbed_prob = pred["probability"]
                        break
            
            # Feature contribution is baseline - probability_without_feature
            # If removing the word drops the probability, the word has POSITIVE contribution
            contribution = baseline_prob - perturbed_prob
            attributions.append((words[i], float(contribution)))
            
        # Softmax normalize contributions for visual presentation
        scores = [attr[1] for attr in attributions]
        max_abs = max(max([abs(s) for s in scores]), 1e-6)
        normalized_attributions = [(attr[0], attr[1] / max_abs) for attr in attributions]
        
        return normalized_attributions
