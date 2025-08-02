from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


import torch


def load_scraped_documents(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    docs = []
    for block in raw.split("URL:")[1:]:
        lines = block.strip().split("\n")
        url = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        docs.append(Document(page_content=content, metadata={"source": url}))
    return docs


def chunk_text(documents, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)



def get_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def store_embeddings(chunks, persist_dir="chroma_db", embedding_model=None):
    db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_dir,
    collection_metadata={"hnsw:space": "cosine"}  # optional
)

    print(f"[✓] Stored {len(chunks)} chunks into vector DB at '{persist_dir}'")
    return db


if __name__ == "__main__":
    docs = load_scraped_documents(r"C:\Users\naver\Desktop\RAG CHATBOT\scraped_content.txt")
    chunks = chunk_text(docs)
    
    embedding_model = get_embedding_model()
    db = store_embeddings(chunks, embedding_model=embedding_model)
