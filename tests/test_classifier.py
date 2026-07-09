import pytest
from src.models.classifier import ZeroShotClassifier
from src.utils.explainability import TextFeatureAttribution

def test_fallback_heuristic_classification():
    classifier = ZeroShotClassifier()
    # Forces mock/fallback mode to run immediately
    res = classifier._mock_classify(
        text="Convolutional Neural Networks are great for computer vision image tasks.",
        candidate_labels=["Computer Vision", "Generative AI", "Time Series"]
    )
    
    assert res["top_label"] == "Computer Vision"
    assert len(res["predictions"]) == 3
    # Check that probabilities sum to 1.0
    probs = [p["probability"] for p in res["predictions"]]
    assert pytest.approx(sum(probs)) == 1.0

def test_feature_attribution_loo():
    classifier = ZeroShotClassifier()
    # Attach custom explainer using classification mock function
    explainer = TextFeatureAttribution(classify_fn=classifier._mock_classify)
    
    text = "mri scan showing brain tumor segments"
    labels = ["Computer Vision", "Generative AI"]
    
    attributions = explainer.explain(text, target_label="Computer Vision", candidate_labels=labels)
    
    # Should return (word, score) for each word
    assert len(attributions) == len(text.split())
    for word, score in attributions:
        assert isinstance(word, str)
        assert -1.0 <= score <= 1.0
