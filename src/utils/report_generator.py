import os
import json

def generate_reports():
    latest_path = "reports/latest.json"
    if not os.path.exists(latest_path):
        print("No latest.json report found. Run benchmark first.")
        return
        
    with open(latest_path, "r") as f:
        report = json.load(f)
        
    md_content = f"""# Benchmark Verification Report: {report['project']}
- **Generated Timestamp**: {report['timestamp']}
- **Status**: {report['status'].upper()}

## Hardware Metadata
- **OS**: {report['hardware']['os']}
- **CPU**: {report['hardware']['cpu']}
- **RAM**: {report['hardware']['ram_gb']} GB
- **GPU**: {report['hardware']['gpu']}

## Environment Versions
- **Python**: {report['environment'].get('python', 'N/A')}
- **PyTorch**: {report['environment'].get('torch', 'N/A')}
- **SentenceTransformers**: {report['environment'].get('sentence_transformers', 'N/A')}

## Model Metadata
- **Model**: {report['metadata']['model']}
- **Parameters**: {report['metadata']['parameters_million']} M
- **Dataset**: {report['metadata']['dataset']}
- **Device**: {report['metadata']['device']}

"""

    if report["status"] == "success":
        b = report["benchmarks"]
        md_content += f"""## Measured Benchmark Results
| Search Strategy | Average Latency | Throughput (QPS) |
| :--- | :---: | :---: |
| **BM25 Sparse** | {b['bm25_latency_ms']} ms | {round(1000 / b['bm25_latency_ms'], 2)} QPS |
| **Dense Embeddings** | {b['dense_latency_ms']} ms | {round(1000 / b['dense_latency_ms'], 2)} QPS |
| **Hybrid Blend** | {b['hybrid_latency_ms']} ms | {b['search_throughput_qps']} QPS |
"""
    else:
        md_content += f"""## Execution Note
- **Reason**: {report.get('reason', 'Unknown reason')}
"""

    with open("reports/latest.md", "w") as f:
        f.write(md_content)
    print("reports/latest.md updated successfully.")

    # 2. Update README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme = f.read()
            
        start_marker = "<!-- BENCHMARK_TABLE_START -->"
        end_marker = "<!-- BENCHMARK_TABLE_END -->"
        
        if start_marker in readme and end_marker in readme:
            if report["status"] == "success":
                b = report["benchmarks"]
                table_md = f"""
| Search Mode | Primary Advantage | Key Trade-off | Measured Latency |
| :--- | :--- | :--- | :---: |
| **BM25 Sparse** | High keyword precision | Misses synonym contexts | {b['bm25_latency_ms']} ms |
| **Dense Embeddings** | Captures semantic context | Weak at exact term matching | {b['dense_latency_ms']} ms |
| **Hybrid Blend** | Combines lexical \& semantic | Requires score normalization | {b['hybrid_latency_ms']} ms |
"""
            else:
                table_md = f"\n*Benchmark Not Run: {report.get('reason', 'Missing dependencies')}*\n"
                
            start_idx = readme.find(start_marker) + len(start_marker)
            end_idx = readme.find(end_marker)
            
            new_readme = readme[:start_idx] + table_md + readme[end_idx:]
            with open(readme_path, "w") as f:
                f.write(new_readme)
            print("README.md benchmark table updated successfully.")
        else:
            print("Benchmark table markers not found in README.md.")

if __name__ == "__main__":
    generate_reports()
