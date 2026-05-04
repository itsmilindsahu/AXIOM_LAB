"""
run_setup.py
------------
One-shot setup: ingest BIS standards, build knowledge graph.
Run this ONCE before launching the Streamlit app or evaluation.

Usage:
    python run_setup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("="*55)
    print("  BIS Standard Discovery — Setup")
    print("="*55)

    print("\n[1/2] Running document ingestion ...")
    try:
        from src.ingestion.ingest import ingest
        ingest()
        print("[Setup] Using BGE sentence-transformer embeddings.")
    except Exception as e:
        print(f"[Setup] BGE model unavailable ({e.__class__.__name__}). "
              "Falling back to offline TF-IDF encoder.")
        from src.ingestion.ingest_offline import ingest_offline
        ingest_offline()

    print("\n[2/2] Building knowledge graph ...")
    from src.graph.knowledge_graph import build_graph, save_graph
    G = build_graph()
    save_graph(G)

    print("\n" + "="*55)
    print("  Setup complete! You can now run:")
    print("    streamlit run app/streamlit_app.py")
    print("    python src/evaluation/evaluate.py --adversarial")
    print("="*55)


if __name__ == "__main__":
    main()
