SYSTEM_PROMPT = (
    "You are a precise scientific assistant. Answer only using the provided context. "
    "If the answer is not contained in the context, say you do not have enough information. "
    "Cite as [page, section] when possible."
)

ANSWER_PROMPT = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Write a concise, well-structured answer grounded in the context."
)
