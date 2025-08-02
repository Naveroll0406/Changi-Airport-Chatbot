# 🛫 Changi Airport AI Chatbot

An AI-powered chatbot that answers user queries about **Changi Airport** and **Jewel Changi** using Retrieval-Augmented Generation (RAG) architecture. Built with LangChain, HuggingFace embeddings, Streamlit UI, and local ChromaDB.

---

## 📂 Project Structure

```
├── embedder.py          # Embeds and stores chunked scraped data into ChromaDB
├── scrapper.py          # Scrapes pages from Changi Airport & Jewel websites
├── rag_chatbot.py       # Core RAG chain logic (retriever + LLM + prompt)
├── rag_chat_ui.py       # Streamlit web interface for interacting with the chatbot
├── chroma_db/           # Directory where vector store is persisted
└── scraped_content.txt  # Scraped web content used for embedding
```

---

## 🧠 Features

- 🔍 **Retrieval-based answers** grounded in scraped airport content
- 🧾 **Metadata filtering** (terminals, services, audience types)
- 💬 **Streaming responses** via LangChain
- 🖼️ **Streamlit UI** for easy interaction
- 🧠 **Few-shot examples** for context-specific answers

---

## 🚀 How to Run

### 1. Scrape Content
```bash
python scrapper.py
```

### 2. Generate Embeddings & Store in ChromaDB
```bash
python embedder.py
```
### 3. Start Ollama (LLM Backend)
```bash
🖥️ Open Command Prompt (CMD) and run the following:

ollama pull mistral   # Pull the model if not already downloaded  
ollama run mistral    # Start the LLM server  


```
### 3. Run the Chatbot (Streamlit UI)
```bash
streamlit run rag_chat_ui.py
```

---

## 🧰 Requirements

- Python 3.8+
- torch
- langchain
- langchain-community
- langchain-core
- langchain-chroma
- langchain-huggingface
- streamlit
- bs4
- selenium
- ollama (for LLM inference like `mistral`)

---

## ⚠️ Notes

- The `chroma_db/chroma.sqlite3` file is not included in this repo due to GitHub's 100 MB file limit.
- Use `scrapper.py` and `embedder.py` to regenerate the database locally.
- LLM is accessed via `Ollama`. Ensure it’s running locally with the required model pulled (`mistral` by default).

---

## 📍 Credits

This assistant is tailored for Changi Airport and Jewel Changi and operates with factual, grounded context.

---
