import streamlit as st
from pathlib import Path
from ingestion.loader import PaperLoader
from ingestion.index import build_index
from llm.chain import answer_question
import atexit, shutil
from config import PERSIST_DIR, ASSETS_DIR

st.set_page_config(page_title="Paper QA", page_icon="📚", layout="wide")
st.title("📚 Research Papers QA")

with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Research paper (PDF)", type=["pdf"])
    if up:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        pdf_path = data_dir / up.name
        pdf_path.write_bytes(up.read())
        st.success(f"Saved: {pdf_path}")

        with st.spinner("Extracting & indexing.."):
            parts = PaperLoader(pdf_path).load()
            build_index(parts["text"], parts["tables"], parts["images"])
        st.success("Index ready ✅")

st.header("2) Ask the paper")
q = st.text_input("Your question:")
if st.button("Answer", disabled=not q):
    with st.spinner("Answering..."):
        ans, docs = answer_question(q)
    st.subheader("Answer")
    st.write(ans)

    st.subheader("Supporting chunks")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        st.markdown(f"**Chunk {i}** — page {meta.get('page')} | section: {meta.get('section')} | type: {meta.get('element_type')}")
        st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))
        if meta.get("table_csv"):
            st.caption(f"Table CSV: {meta['table_csv']}")
        if meta.get("image_path"):
            st.image(meta["image_path"])

def cleanup():
    try:
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
        if ASSETS_DIR.exists():
            shutil.rmtree(ASSETS_DIR)
        print("✅ All vectors, tables, and images have been deleted after app exit.")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

atexit.register(cleanup)

