📑 Research Paper Q&A with RAG
🔍 Introduction
This project implements a Retrieval-Augmented Generation (RAG) system to:

Extract content from research papers (PDFs).

Normalize, chunk, and store the content in a vector database (ChromaDB).

Perform retrieval + re-ranking (using a Haystack Ranker).

Generate answers using LLaMA (via HuggingFace or Ollama).

Integrate the system into an intuitive Streamlit interface.

This application helps students and researchers quickly summarize and ask questions about the content of scientific papers.

⚙️ System Architecture
The pipeline is structured as follows:

Extract & Preprocess

Uses UnstructuredPDFLoader to extract text, tables, and images from PDFs.

Vectorization & Storage

Embeddings are created using a HuggingFace model (sentence-transformers/all-MiniLM-L6-v2).

Vectors are stored in ChromaDB (persist_directory).

Retrieval & Ranking

Retriever: A Chroma retriever with top_k=20.

Ranker: A Haystack SentenceTransformersDiversityRanker (a cross-encoder, such as ms-marco-MiniLM-L-6-v2).

Answer Generation

The re-ranked context is provided to the prompt of the LLaMA model.

The model generates a response in a scientific assistant style.

UI Layer

Streamlit is used for uploading PDFs, entering questions, and displaying results.

After the application is closed, all data (vectors, tables, and images) are automatically deleted to ensure data privacy.

📦 Installation
Requirements
Python 3.10+

RAM >= 8 GB (16 GB recommended)

Dependencies
Clone the repository and install the dependencies:

Bash

git clone https://github.com/honglinhvt19/Paper-QA.git
cd Paper-QA
pip install -r requirements.txt
requirements.txt
streamlit
langchain
langchain-community
transformers
torch
sentence-transformers
chromadb
farm-haystack
unstructured
pypdf
Models
Embeddings: sentence-transformers/all-MiniLM-L6-v2

Ranker: cross-encoder/ms-marco-MiniLM-L-6-v2

LLM:

Local: Ollama (llama3.1:8b)

HuggingFace Hub (for users with a more powerful GPU)

🚀 How to Run
Start the app

Bash

streamlit run app.py
Access
Open your browser and navigate to: http://localhost:8501

💻 Usage
Upload a research paper in PDF format.

The system will:

Extract, chunk, and index the content into ChromaDB.

The retriever will fetch the top 30 relevant chunks.

The ranker will re-rank and select the best 5 chunks.

LLaMA will generate the final answer.

View the generated answer and the supporting chunks.

All vector data, tables, and images are automatically deleted when the app is closed.

📊 Demo Q&A
Example with the paper Xception: Deep Learning with Depthwise Separable Convolutions (Chollet, 2017):

Q: What improvement does Xception make over Inception V3?

A: Xception replaces Inception modules with depthwise separable convolutions, which makes better use of parameters and improves generalization.

🔮 Roadmap
[ ] Support for multiple papers simultaneously.

[ ] Add a domain-specific ranker (e.g., SciBERT cross-encoder).

[ ] Allow exporting answers to PDF/Markdown.

[ ] Integrate multi-modal support (text + tables + images).

🤝 Contributions
All contributions and pull requests are welcome. Feel free to open an issue to discuss.

📜 License
MIT License © 2025