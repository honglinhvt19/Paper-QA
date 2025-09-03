from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from .prompts import SYSTEM_PROMPT, ANSWER_PROMPT
from .backend import get_llm
from retrieval.chroma_search import retrieve_initial
from retrieval.haystack_rerank import rerank

def _format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        sec = d.metadata.get("section") or ""
        typ = d.metadata.get("element_type", "text")
        head = f"[page {page}{', ' + sec if sec else ''} | {typ}]"
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n".join(parts)

def answer_question(question: str):
    initial = retrieve_initial(question)
    top_docs = rerank(question, initial)

    tmpl = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", ANSWER_PROMPT),
    ])
    messages = tmpl.format_messages(question=question, context=_format_context(top_docs))
    llm = get_llm()
    resp = llm.invoke(messages)
    return resp.content, top_docs
