from config import CHROMA_TOP_K
from retrieval.vectorstore import get_vectorstore

def retrieve_initial(query: str, k: int = CHROMA_TOP_K):
    db, _ = get_vectorstore()
    return db.similarity_search(query, k=k)
