# Model Card: DocMind AI NLP Engines

This model card details the specification and capabilities of the models integrated into the **DocMind AI** intelligent search platform.

## Model Details
- **Developer**: Portfolio Owner
- **Model Types**:
  - **SentenceTransformer**: Encoder model (`sentence-transformers/all-MiniLM-L6-v2`) mapped to extract dense textual embeddings.
  - **Cross-Encoder**: Sequence pairwise classifier (`cross-encoder/ms-marco-MiniLM-L-6-v2`) used to rerank dense/sparse candidate results.
  - **Zero-Shot Classifier**: Dynamic sequence classifier (`facebook/bart-large-mnli` or similar baseline) to assign texts to categories without custom training.
- **Task**: Information Retrieval, Hybrid Text Search, Multi-label Sequence Classification.

## Intended Use
- **Primary Intended Use**: Portfolio presentation, academic text search index evaluations, and demonstrating zero-shot explainability mappings.
- **Out of Scope**: High-risk filtering (e.g. medical triage or automated legal validation) without human review.

## Training & Evaluation Datasets
- Evaluated on sample corpora consisting of diverse AI/ML domain topics. 
- Custom synonyms are used in the semantic expansion phase to increase vocabulary coverage.
- Custom Leave-One-Out (LOO) attributions are used to explain category decisions at a word/token-level.
