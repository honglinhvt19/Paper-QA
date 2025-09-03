from typing import List
import uuid
from PIL import Image as PILImage
from langchain.schema import Document
from config import ASSETS_DIR
from retrieval.vectorstore import get_vectorstore

def _save_table_csv(table_obj, name: str) -> str:
    csv_path = ASSETS_DIR / f"{name}.csv"
    try:
        if hasattr(table_obj, "to_pandas"):
            table_obj.to_pandas().to_csv(csv_path, index=False)
        elif hasattr(table_obj, "to_csv"):
            table_obj.to_csv(str(csv_path))
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(getattr(table_obj, "text", "") or "")
    except Exception:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(getattr(table_obj, "text", "") or "")
    return str(csv_path)

def _save_image(image_el, name: str) -> str | None:
    try:
        img = getattr(image_el, "image", None) 
        if isinstance(img, PILImage.Image):
            out = ASSETS_DIR / f"{name}.png"
            img.save(out)
            return str(out)
    except Exception:
        return None
    return None

def build_index(text_docs: List[Document], table_docs: List[Document], image_docs: List[Document]):
    processed: List[Document] = []

    for i, d in enumerate(table_docs):
        tbl = d.metadata.pop("_table", None)
        if tbl is not None:
            d.metadata["table_csv"] = _save_table_csv(tbl, f"table_{i}_{uuid.uuid4().hex[:6]}")
        if not d.page_content:
            d.page_content = "[TABLE]"
        processed.append(d)

    for i, d in enumerate(image_docs):
        img = d.metadata.pop("_image", None)
        if img is not None:
            p = _save_image(img, f"image_{i}_{uuid.uuid4().hex[:6]}")
            if p:
                d.metadata["image_path"] = p
        processed.append(d)

    processed.extend(text_docs)

    vectordb, embedder = get_vectorstore(create_if_missing=True)
    vectordb.add_documents(processed)
    vectordb.persist()
