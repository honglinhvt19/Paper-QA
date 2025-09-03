from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import PERSIST_DIR, EMBEDDING_MODEL

_embedder = None
_vectordb = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder

def get_vectorstore(create_if_missing: bool = False):
    global _vectordb
    emb = get_embedder()
    if _vectordb is None:
        _vectordb = Chroma(
            collection_name="paper_chunks",
            persist_directory=str(PERSIST_DIR),
            embedding_function=emb,
        )
        if create_if_missing:
            _vectordb.persist()
    return _vectordb, emb
