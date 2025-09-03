from typing import List
from haystack.components.rankers import SentenceTransformersDiversityRanker
from haystack import Document as HayDoc
from langchain.schema import Document
from config import RERANK_MODEL, RERANK_TOP_K

_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = SentenceTransformersDiversityRanker(model=RERANK_MODEL, similarity="cosine")
        _ranker.warm_up()
    return _ranker

def rerank(query: str, lc_docs: List[Document]) -> List[Document]:
    ranker = get_ranker()
    hay_docs = [HayDoc(content=d.page_content, meta=d.metadata) for d in lc_docs]
    ranked = ranker.run(query=query, documents=hay_docs, top_k=RERANK_TOP_K)
    docs = ranked.get("documents", [])
    out = [Document(page_content=d.content, metadata=d.meta or {}) for d in docs]
    return out
