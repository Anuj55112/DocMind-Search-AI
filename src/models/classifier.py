import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ZeroShotClassifier:
    """
    Wrapper around HuggingFace's zero-shot classification pipeline.
    Enables multi-class dynamic classification without target model training.
    """
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.pipeline = None
        self.initialized = False

    def initialize(self) -> bool:
        if self.initialized:
            return True
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading zero-shot classification pipeline using {self.model_name}...")
            self.pipeline = pipeline("zero-shot-classification", model=self.model_name, device=device)
            self.initialized = True
            logger.info("Zero-shot classifier initialized successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load zero-shot classifier: {e}. Classification will run in mock mode.")
            return False

    def classify(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """
        Classifies input text into one of the candidate labels.
        """
        if not self.initialized:
            success = self.initialize()
            if not success or self.pipeline is None:
                return self._mock_classify(text, candidate_labels)

        try:
            result = self.pipeline(text, candidate_labels=candidate_labels)
            # Reformat to sorted labels with probabilities
            labels = result["labels"]
            scores = result["scores"]
            
            predictions = []
            for label, score in zip(labels, scores):
                predictions.append({
                    "label": label,
                    "probability": float(score)
                })
                
            return {
                "text": text,
                "top_label": labels[0],
                "predictions": predictions,
                "engine": "HuggingFace Zero-Shot"
            }
        except Exception as e:
            logger.error(f"Error during zero-shot classification: {e}")
            return self._mock_classify(text, candidate_labels)

    def _mock_classify(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Local keyword-based fallback classifier when model download fails."""
        text_lower = text.lower()
        scores = {}
        
        # Simple heuristic mappings
        keywords = {
            "Computer Vision": ["vision", "image", "segment", "cnn", "pixel", "yolo", "mri", "x-ray", "sam"],
            "Generative AI": ["llm", "gpt", "rag", "generate", "chatbot", "prompt", "token", "hallucination"],
            "Time Series": ["series", "forecast", "predict", "tft", "lstm", "prophet", "demand", "stock"],
            "Information Retrieval": ["search", "retrieve", "bm25", "vector", "embedding", "similarity", "index", "rerank"]
        }
        
        for label in candidate_labels:
            score = 0.1 # Base uniform probability
            words = keywords.get(label, [])
            for word in words:
                if word in text_lower:
                    score += 0.3
            scores[label] = score
            
        # Normalize scores to form a probability distribution
        total = sum(scores.values())
        predictions = []
        for label, val in scores.items():
            predictions.append({
                "label": label,
                "probability": float(val / total)
            })
            
        predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        return {
            "text": text,
            "top_label": predictions[0]["label"],
            "predictions": predictions,
            "engine": "Heuristic Mock Engine"
        }
