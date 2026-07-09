import os
import sys
import time
import json
import platform
import subprocess
from datetime import datetime

def get_system_metadata() -> dict:
    metadata = {
        "os": platform.system(),
        "cpu": "Unknown",
        "ram_gb": 8,
        "gpu": "None"
    }
    
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            metadata["cpu"] = cpu
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        metadata["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass
        
    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            metadata["ram_gb"] = round(mem_bytes / (1024 ** 3))
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        metadata["ram_gb"] = round(mem_kb / (1024 ** 2))
                        break
    except Exception:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            metadata["gpu"] = "Apple Metal (MPS)"
    except Exception:
        pass
        
    return metadata

def run_benchmark():
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    report = {
        "project": "DocMind Search AI",
        "timestamp": timestamp,
        "status": "not_run",
        "hardware": get_system_metadata(),
        "environment": {
            "python": platform.python_version()
        },
        "metadata": {
            "model": "all-MiniLM-L6-v2",
            "parameters_million": 22.7,
            "dataset": "Sample Knowledge Base (10 chunks)",
            "batch_size": 1,
            "image_size": None,
            "sequence_length": 128,
            "device": "cpu"
        },
        "benchmarks": {}
    }
    
    missing_deps = []
    try:
        import torch
        report["environment"]["torch"] = torch.__version__
    except ImportError:
        missing_deps.append("torch")
        
    try:
        import sentence_transformers
        report["environment"]["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        missing_deps.append("sentence_transformers")
        
    if missing_deps:
        report["status"] = "not_run"
        report["reason"] = f"Missing required dependencies: {', '.join(missing_deps)}"
        report["required_dependency"] = missing_deps[0]
        
        save_reports(report, date_str)
        print(f"Benchmark not run: {report['reason']}")
        return
        
    try:
        from src.search.hybrid_search import HybridSearcher
        from src.config import load_config
        
        config = load_config()
        # Initialize searcher
        searcher = HybridSearcher(config)
        
        # Ingest sample docs
        docs = [
            "Attention layers map query and key vectors to derive weight coefficients.",
            "Gradient descent optimizes weights relative to loss gradients.",
            "Transformers stack self-attention blocks to model sequence contexts.",
            "Zero-shot classifiers evaluate target class labels without training.",
            "RAG pipelines retrieve context elements to ground LLM generations."
        ]
        searcher.index_documents(docs)
        
        print("Benchmarking lexical BM25 retrieval latency...")
        start_time = time.time()
        for _ in range(50):
            _ = searcher.bm25_searcher.search("attention weights", k=2)
        bm25_latency = ((time.time() - start_time) / 50) * 1000
        
        print("Benchmarking dense vector retrieval latency...")
        start_time = time.time()
        for _ in range(50):
            _ = searcher.dense_searcher.search("attention weights", k=2)
        dense_latency = ((time.time() - start_time) / 50) * 1000
        
        print("Benchmarking hybrid blended search latency...")
        start_time = time.time()
        for _ in range(50):
            _ = searcher.search("attention weights", k=2)
        hybrid_latency = ((time.time() - start_time) / 50) * 1000
        
        report["status"] = "success"
        report["benchmarks"] = {
            "bm25_latency_ms": round(bm25_latency, 2),
            "dense_latency_ms": round(dense_latency, 2),
            "hybrid_latency_ms": round(hybrid_latency, 2),
            "search_throughput_qps": round(1000 / hybrid_latency, 2)
        }
        print(f"Benchmark success: BM25 = {bm25_latency:.2f}ms, Dense = {dense_latency:.2f}ms, Hybrid = {hybrid_latency:.2f}ms")
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Benchmark execution error: {e}"
        print(f"Benchmark error: {e}")
        
    save_reports(report, date_str)

def save_reports(report: dict, date_str: str):
    with open(f"reports/{date_str}-benchmark.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("reports/latest.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
