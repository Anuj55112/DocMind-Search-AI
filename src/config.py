import os
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # Model config
        model = config_dict.get("model", {})
        self.encoder_model_name: str = model.get("encoder_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.cross_encoder_model_name: str = model.get("cross_encoder_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.classifier_model_name: str = model.get("classifier_model_name", "distilbert-base-uncased")
        
        # Search config
        search = config_dict.get("search", {})
        self.top_k: int = search.get("top_k", 5)
        self.hybrid_alpha: float = search.get("hybrid_alpha", 0.5)
        self.query_expansion: bool = search.get("query_expansion", True)
        
        # API config
        api = config_dict.get("api", {})
        self.api_host: str = api.get("host", "0.0.0.0")
        self.api_port: int = api.get("port", 8001)

def load_config(config_path: str = "") -> Config:
    if not config_path:
        current_dir = Path(__file__).resolve().parent
        config_path = str(current_dir.parent / "configs" / "config.yaml")
        
    if not os.path.exists(config_path):
        return Config({})
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    return Config(config_dict)
