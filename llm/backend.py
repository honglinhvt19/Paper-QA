from config import LLM_BACKEND, OLLAMA_MODEL, HF_LLAMA, HF_DEVICE

def get_llm():
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain.llms import HuggingFacePipeline
        tok = AutoTokenizer.from_pretrained(HF_LLAMA)
        model = AutoModelForCausalLM.from_pretrained(HF_LLAMA, device_map=HF_DEVICE)
        gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=gen)
