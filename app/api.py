import os
import json
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Package imports
from src.config import load_config
from src.search.hybrid_search import HybridSearchEngine
from src.models.classifier import ZeroShotClassifier
from src.utils.explainability import TextFeatureAttribution

app = FastAPI(
    title="DocMind AI - Intelligent Search API",
    description="REST API for Hybrid search (BM25 + SentenceTransformers), Cross-Encoder reranking, and Zero-shot classification with LOO token attribution explanations.",
    version="1.0.0"
)

# Load config
config = load_config()

# Global states
CORPUS: List[Dict[str, str]] = []
SEARCH_ENGINE: Optional[HybridSearchEngine] = None
CLASSIFIER: Optional[ZeroShotClassifier] = None
EXPLAINER: Optional[TextFeatureAttribution] = None

# Input schemas
class ClassifyRequest(BaseModel):
    text: str
    candidate_labels: List[str]

class ExplainRequest(BaseModel):
    text: str
    target_label: str
    candidate_labels: List[str]

class QARequest(BaseModel):
    query: str
    top_k: int = 3

@app.on_event("startup")
def startup_event():
    global CORPUS, SEARCH_ENGINE, CLASSIFIER, EXPLAINER
    
    # Load corpus
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample", "documents.json"))
    if os.path.exists(corpus_path):
        try:
            with open(corpus_path, "r") as f:
                CORPUS = json.load(f)
            print(f"Loaded corpus containing {len(CORPUS)} documents.")
        except Exception as e:
            print(f"Error loading corpus JSON: {e}")
            CORPUS = []
            
    if not CORPUS:
        # Fallback default documents if loading fails
        CORPUS = [
            {"id": "d1", "title": "Vision", "content": "Computer Vision is cnn image classification.", "category": "Vision"},
            {"id": "d2", "title": "Generative", "content": "Generative AI uses LLMs and RAG workflows.", "category": "GenAI"}
        ]
        
    # Instantiate search engine
    SEARCH_ENGINE = HybridSearchEngine(
        corpus=CORPUS,
        encoder_model_name=config.encoder_model_name,
        cross_encoder_model_name=config.cross_encoder_model_name,
        hybrid_alpha=config.hybrid_alpha,
        query_expansion=config.query_expansion
    )
    
    # Instantiate zero-shot classifier
    CLASSIFIER = ZeroShotClassifier(model_name=config.classifier_model_name)
    
    # Instantiate explainer
    EXPLAINER = TextFeatureAttribution(classify_fn=CLASSIFIER.classify)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "corpus_size": len(CORPUS),
        "search_engine_ready": SEARCH_ENGINE is not None,
        "classifier_ready": CLASSIFIER is not None
    }

@app.get("/search")
def run_search(
    query: str = Query(..., description="The query string to search for"),
    top_k: int = Query(5, description="Number of results to return")
):
    if not SEARCH_ENGINE:
        raise HTTPException(status_code=500, detail="Search engine not initialized.")
    try:
        results = SEARCH_ENGINE.search(query, top_k=top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.post("/classify")
def run_classification(request: ClassifyRequest):
    if not CLASSIFIER:
        raise HTTPException(status_code=500, detail="Classifier not initialized.")
    try:
        result = CLASSIFIER.classify(request.text, request.candidate_labels)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.post("/explain")
def run_explanation(request: ExplainRequest):
    if not EXPLAINER:
        raise HTTPException(status_code=500, detail="Explainer not initialized.")
    try:
        attributions = EXPLAINER.explain(request.text, request.target_label, request.candidate_labels)
        return {
            "text": request.text,
            "target_label": request.target_label,
            "attributions": attributions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation calculation failed: {e}")

@app.post("/summary")
def generate_summary(request: QARequest):
    """
    Multi-document QA Summarization.
    Retrieves the top_k matching documents and generates an intelligence summary.
    """
    if not SEARCH_ENGINE:
        raise HTTPException(status_code=500, detail="Search engine not initialized.")
        
    try:
        results = SEARCH_ENGINE.search(request.query, top_k=request.top_k)
        if not results:
            return {"query": request.query, "summary": "No matching context found to summarize.", "sources": []}
            
        # Combine retrieved context
        context_blocks = []
        sources = []
        for r in results:
            doc = r["document"]
            context_blocks.append(f"Title: {doc['title']}\nContent: {doc['content']}")
            sources.append(doc["title"])
            
        combined_context = "\n\n".join(context_blocks)
        
        # Heuristic Summarizer (mimicking extractive LLM QA response for instant offline local setup)
        # In production this parses to OpenAI API/Ollama
        summary_intro = f"Based on the retrieved sources ({', '.join(sources[:2])}), here is the synthesis:\n\n"
        synthesis = ""
        for r in results:
            doc = r["document"]
            synthesis += f"- Regarding **{doc['category']}**: {doc['content'][:120]}...\n"
            
        summary = summary_intro + synthesis + "\nThis overview addresses key aspects of your query: " + request.query
        
        return {
            "query": request.query,
            "summary": summary,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=config.api_host, port=config.api_port, reload=True)
