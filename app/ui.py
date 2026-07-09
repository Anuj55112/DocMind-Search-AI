import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="DocMind AI - Enterprise Search Platform",
    page_icon="🔍",
    layout="wide"
)

# Premium stylesheet matching portfolio guidelines
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #8A8F98;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background-color: #171B26;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2B303C;
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00F2FE;
    }
    
    .highlight-box {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #1A1F2C;
        border-left: 4px solid #00F2FE;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    /* Token highlighting classes */
    .token-pos {
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 2px;
        display: inline-block;
    }
    .token-neg {
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint definition
API_URL = os.getenv("DOCMIND_API_URL", "http://localhost:8001")

# Mocks in case server is loading weights
def mock_search(query):
    return [
        {
            "document": {"title": "Mock ResNet Paper", "content": f"ResNet is a residual CNN architecture. Query match for: {query}", "category": "Computer Vision"},
            "bm25_score": 1.5, "dense_score": 0.8, "hybrid_score": 0.95, "rerank_score": 1.8
        }
    ]

def mock_classify(text, labels):
    predictions = [{"label": labels[0], "probability": 0.75}, {"label": labels[1], "probability": 0.25}]
    return {"text": text, "top_label": labels[0], "predictions": predictions}

def mock_explain(text, target_label, labels):
    words = text.split()
    attributions = []
    for i, w in enumerate(words):
        # Assign mock sine scores
        score = float(np.sin(i))
        attributions.append([w, score])
    # Normalize
    scores = [a[1] for a in attributions]
    max_abs = max(max([abs(s) for s in scores]), 1e-6)
    normalized = [[a[0], a[1]/max_abs] for a in attributions]
    return {"text": text, "target_label": target_label, "attributions": normalized}

def mock_summary(query):
    return {
        "summary": "This is a synthesized summary matching: " + query + ". Context retrieved covers deep networks and classification structures.",
        "sources": ["Mock ResNet Paper"]
    }

# Header Section
col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<div class='main-title'>DocMind AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Intelligent Enterprise Search, Hybrid Retrieval, & Explained Classification</div>", unsafe_allow_html=True)
with col2:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            st.success("API: Connected")
        else:
            st.warning("API: Loading Models")
    except Exception:
        st.error("API: Offline (Using Fallback)")

# Tabs
tab1, tab2 = st.tabs(["🔍 Intelligent Search", "🏷️ Topic Classifier & SHAP Explain"])

# TAB 1: INTELLIGENT SEARCH
with tab1:
    st.markdown("### Enterprise Search Console")
    col_input, col_results = st.columns([4, 6])
    
    with col_input:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Search Queries")
        query_input = st.text_input("Enter your research topic or question...", "How does RAG solve LLM hallucinations?")
        top_k = st.slider("Documents to Retrieve", 1, 6, 3)
        st.markdown("</div>", unsafe_allow_html=True)
        
        btn_search = st.button("⚡ Execute Search", use_container_width=True)
        
    with col_results:
        if btn_search:
            with st.spinner("Searching document index..."):
                # Fetch search results
                try:
                    r = requests.get(f"{API_URL}/search", params={"query": query_input, "top_k": top_k}, timeout=10)
                    results = r.json().get("results", [])
                except Exception:
                    results = mock_search(query_input)
                    
                # Fetch summary QA
                try:
                    r = requests.post(f"{API_URL}/summary", json={"query": query_input, "top_k": top_k}, timeout=15)
                    summary_data = r.json()
                except Exception:
                    summary_data = mock_summary(query_input)
                    
                # Display Summary QA
                st.markdown("#### 💡 AI Synthesized QA Summary")
                st.markdown(f"<div class='highlight-box'>{summary_data['summary']}</div>", unsafe_allow_html=True)
                
                # Display Documents list
                st.markdown("#### 📄 Matching Documents")
                for i, res in enumerate(results):
                    doc = res["document"]
                    st.markdown(f"""
                    <div class='section-card'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>{i+1}. {doc['title']}</strong>
                            <span style='background-color: #2B303C; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;'>{doc['category']}</span>
                        </div>
                        <p style='color: #CFD2D6; margin-top: 0.8rem; font-size: 0.95rem;'>{doc['content']}</p>
                        <hr style='border: 1px solid #2B303C; margin: 0.6rem 0;'>
                        <div style='display: flex; gap: 1.5rem; font-size: 0.8rem; color: #8A8F98;'>
                            <span>BM25 Score: {res['bm25_score']:.2f}</span>
                            <span>Dense Similarity: {res['dense_score']:.2f}</span>
                            <span>Hybrid Score: {res['hybrid_score']:.2f}</span>
                            <span>Rerank Score: {res.get('rerank_score', 0.0):.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Input a search query and execute search to display relevant research details and RAG summaries.")

# TAB 2: TOPIC CLASSIFIER & SHAP EXPLAIN
with tab2:
    st.markdown("### Zero-Shot Classifier & Word Attribution (SHAP)")
    col_cls_input, col_cls_results = st.columns([4, 6])
    
    with col_cls_input:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Input Text")
        text_input = st.text_area(
            "Paragraph to classify...",
            "The Segment Anything Model (SAM) allows zero-shot interactive segmentation using point and box coordinates. It is great for computer vision MRI processing.",
            height=150
        )
        labels_input = st.text_input(
            "Candidate Categories (comma separated)",
            "Computer Vision, Generative AI, Time Series, Information Retrieval"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        btn_classify = st.button("🏷️ Run Classifier", use_container_width=True)
        
    with col_cls_results:
        if btn_classify:
            candidate_labels = [lbl.strip() for lbl in labels_input.split(",")]
            
            with st.spinner("Classifying text and evaluating word attributions..."):
                # Classification request
                try:
                    r = requests.post(f"{API_URL}/classify", json={"text": text_input, "candidate_labels": candidate_labels}, timeout=15)
                    cls_res = r.json()
                except Exception:
                    cls_res = mock_classify(text_input, candidate_labels)
                    
                top_class = cls_res["top_label"]
                
                # Explanation request
                try:
                    r = requests.post(f"{API_URL}/explain", json={"text": text_input, "target_label": top_class, "candidate_labels": candidate_labels}, timeout=20)
                    explain_res = r.json()
                except Exception:
                    import numpy as np
                    explain_res = mock_explain(text_input, top_class, candidate_labels)
                    
                # Present predictions chart
                st.markdown(f"#### Topic Probability (Top: **{top_class}**)")
                chart_df = pd.DataFrame(cls_res["predictions"])
                fig = px.bar(
                    chart_df, 
                    x="probability", 
                    y="label", 
                    orientation='h',
                    color="probability",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(showlegend=False, paper_bgcolor="#0E1117", plot_bgcolor="#171B26", font_color="#FFFFFF", height=220)
                st.plotly_chart(fig, use_container_width=True)
                
                # Render Highlighted text representing feature attribution
                st.markdown(f"#### Token Feature Attribution (Target: **{top_class}**)")
                
                highlighted_html = ""
                for word, score in explain_res["attributions"]:
                    # Positive score => Cyan (rgba(0, 242, 254, opacity))
                    # Negative score => Magenta (rgba(255, 0, 127, opacity))
                    opacity = min(abs(score), 1.0) * 0.7
                    if score > 0.05:
                        bg_color = f"rgba(0, 242, 254, {opacity})"
                        text_color = "#000000" if opacity > 0.4 else "#FFFFFF"
                        highlighted_html += f"<span class='token-pos' style='background-color: {bg_color}; color: {text_color};'>{word}</span>"
                    elif score < -0.05:
                        bg_color = f"rgba(255, 0, 127, {opacity})"
                        text_color = "#FFFFFF"
                        highlighted_html += f"<span class='token-neg' style='background-color: {bg_color}; color: {text_color};'>{word}</span>"
                    else:
                        highlighted_html += f" <span>{word}</span> "
                        
                st.markdown(f"<div style='background-color: #171B26; padding: 1.5rem; border-radius: 12px; border: 1px solid #2B303C; line-height: 2;'>{highlighted_html}</div>", unsafe_allow_html=True)
                
                st.markdown("""
                <div style='font-size: 0.8rem; color: #8A8F98; margin-top: 0.6rem; display: flex; gap: 1.5rem;'>
                    <span>Legend: <span style='background-color: rgba(0,242,254,0.4); padding: 1px 4px; border-radius: 3px; color: #00F2FE;'>Cyan (Positive)</span> supports class decision.</span>
                    <span><span style='background-color: rgba(255,0,127,0.4); padding: 1px 4px; border-radius: 3px; color: #FF007F;'>Magenta (Negative)</span> contradicts class decision.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Input a paragraph and click 'Run Classifier' to see topic distributions and SHAP word attribution overlays.")
