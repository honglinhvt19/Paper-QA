from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PERSIST_DIR = DATA_DIR / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = DATA_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
CHROMA_TOP_K = 20
RERANK_TOP_K = 6
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM backend:
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "llama3.1:8b"

HF_LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_DEVICE = "cpu"  # "cuda" nếu có GPU

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
