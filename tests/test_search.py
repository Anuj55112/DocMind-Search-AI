import pytest
from src.search.hybrid_search import SimpleBM25, HybridSearchEngine

def test_custom_bm25_scoring():
    corpus = [
        {"id": "1", "title": "A", "content": "Information retrieval and search engines."},
        {"id": "2", "title": "B", "content": "Convolutional Neural Networks for vision tasks."}
    ]
    bm25 = SimpleBM25(corpus)
    
    # "search" keyword should rank doc 1 much higher than doc 2
    scores = bm25.score("search")
    assert scores[0] > scores[1]
    assert scores[1] == 0.0

def test_query_expansion():
    corpus = [{"id": "1", "title": "A", "content": "test text"}]
    engine = HybridSearchEngine(corpus)
    
    # "vision" should expand with synonyms
    expanded = engine.expand_query("vision")
    assert "image" in expanded
    assert "cnn" in expanded

def test_hybrid_search_fallback():
    corpus = [
        {"id": "1", "title": "A", "content": "Information retrieval and search engines."},
        {"id": "2", "title": "B", "content": "Convolutional Neural Networks for vision tasks."}
    ]
    # Initialize engine
    engine = HybridSearchEngine(corpus, query_expansion=False)
    results = engine.search("retrieval", top_k=2)
    
    assert len(results) == 2
    # The first document contains "retrieval" and should have a higher score
    assert results[0]["document"]["id"] == "1"
