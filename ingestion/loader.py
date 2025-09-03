from pathlib import Path
from typing import Dict, List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image
from langchain.schema import Document

def _clean(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r", "").strip()
    return t

class PaperLoader:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

    def load(self) -> Dict[str, List[Document]]:
        elements = partition_pdf(
            filename=str(self.pdf_path),
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
        )
        text_docs, table_docs, image_docs = [], [], []
        current_section = None

        for el in elements:
            meta = {
                "source": str(self.pdf_path),
                "page": getattr(el.metadata, "page_number", None),
                "element_type": el.category,
                "section": current_section,
            }

            if el.category in {"Title", "Header", "Heading"} and getattr(el, "text", None):
                current_section = _clean(el.text) or current_section

            if getattr(el, "text", None) and el.category in {"NarrativeText", "Text"}:
                content = _clean(el.text)
                if content:
                    text_docs.append(Document(page_content=content, metadata=meta))

            elif isinstance(el, Table):
                content = _clean(getattr(el, "text", "") or "[TABLE]")
                d = Document(page_content=content, metadata=meta | {"_table": el})
                table_docs.append(d)

            elif isinstance(el, Image):
                d = Document(page_content="[IMAGE]", metadata=meta | {"_image": el})
                image_docs.append(d)

        return {"text": text_docs, "tables": table_docs, "images": image_docs}
