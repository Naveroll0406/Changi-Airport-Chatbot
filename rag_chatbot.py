""" here's your fully updated code with the memory-aware follow-up fix 🔥. 
I haven’t changed your existing structure — only added the patch exactly where needed.

"""

"""
All your existing features:

✅ Streaming output  
✅ Metadata filtering  
✅ Few-shot prompting  
✅ Terminal/Jewel info  
✅ Keyword awareness  
✅ Follow-up memory handling ✅  
"""

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# 🔧 STEP 1: VectorStore
# model_name="intfloat/e5-small-v2" --- faster model

def get_vectorstore(persist_dir="chroma_db", model="BAAI/bge-small-en-v1.5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = HuggingFaceEmbeddings(
        model=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True,"batch_size":32}
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)

# 🔧 STEP 2: LLM
def get_llm(model_name="mistral", temperature=0.3):
    return OllamaLLM(model=model_name, temperature=temperature, streaming=True)

# 🔧 STEP 3: Few-shot Prompt
def get_combiner(llm):
    examples = [
        {
            "input": "What services are available at Jewel Changi?",
            "context": "Jewel Changi offers attractions like the Rain Vortex and Canopy Park, along with shopping, dining, and lounges.",
            "answer": "Jewel Changi offers the Rain Vortex, Canopy Park, shopping, dining, and lounge access for travelers."
        },
        {
            "input": "Is there a hotel inside Changi Airport?",
            "context": "YOTELAIR is located in Jewel Changi. Aerotel Singapore is in Terminal 1. Both cater to short-stay and transit passengers.",
            "answer": "Yes, Changi Airport has hotels like YOTELAIR in Jewel and Aerotel in Terminal 1 for short stays."
        },
        {
            "input": "How can I reach Terminal 1 from Jewel?",
            "context": "Jewel is connected to Terminal 1 via link bridges. It's directly accessible through Level 1 Arrival Hall.",
            "answer": "You can walk to Terminal 1 from Jewel via the link bridges at Level 1 Arrival Hall."
        },
        {
            "input": "Where can I find restrooms in Terminal 1?",
            "context": "Restrooms are near Gates A–G on Level 1 and 3. Family restrooms are near Gates A1, B1, and F1.",
            "answer": "In Terminal 1, restrooms are near Gates A–G on Levels 1 and 3. Family restrooms are near Gates A1, B1, and F1."
        },
        {
            "input": "Are there any promotions at Changi Airport?",
            "context": "Changi Airport frequently runs retail and dining promotions. Spend S$60 and win concert tickets, for example.",
            "answer": "Yes, Changi Airport often has retail and dining promotions. For instance, spending S$60 may get you concert tickets."
        },
        {
            "input": "What can I do if my flight is delayed?",
            "context": "Travelers can enjoy lounges, themed gardens, shopping, or free movies at ST3PS during delays.",
            "answer": "If your flight is delayed, you can visit lounges, themed gardens, shop, or enjoy free movies at ST3PS."
        },
        {
            "input": "Can I use public transport to get to Changi?",
            "context": "Changi Airport is well-connected via MRT (Changi Airport station), buses, and taxis.",
            "answer": "Yes, you can use MRT, buses, or taxis to reach Changi Airport easily."
        }
    ]

    system_prefix = (
        "You are a professional assistant specialized in answering questions about Changi Airport and Jewel Changi.\n"
        "Follow these strict rules when answering:\n"
        "1. Use only the provided context.\n"
        "2. Do not make up facts or hallucinate.\n"
        "3. Keep answers detailed, clear, and informative.\n"
        "4. No speculation — if unsure, say you don’t know.\n"
        "5. Never mention context or metadata in the reply.\n"
        "6. Use airport-specific language (gates, terminals, lounges).\n"
        "7. Avoid repetition — be concise.\n"
        "9. Mention terminal names or services precisely when needed.\n"
        "10. If context lacks the answer, reply with:\n"
        "    'Sorry, I couldn’t find that information right now.'\n"
        "11. Always remain helpful and neutral in tone.\n"
        "12. Never refer to yourself or your knowledge base.\n"
        "13. Don’t say 'based on the context' or similar phrases.\n"
        "14. Stay strictly factual and airport-focused.\n"
        "15. Never answer outside the scope of Changi Airport or Jewel.\n"
        "16. If the user asks a follow-up, use previous questions to understand the context.\n"
        "17. Do not repeat the same information across different questions.\n"
        "18. If a question is unclear, ask for clarification instead of guessing.\n"
        "19. Mention the terminal name explicitly when describing locations.\n"
        "20. Avoid using bullet points unless listing multiple services.\n"
        "21. Keep answers clear and informative. Be Descriptive on answer and give details or context, provide richer and longer descriptions.\n"
        "22. Do not mention app names like 'iShopChangi' unless the user explicitly asks.\n"
        "23. Avoid promotional or marketing language — stay factual and functional.\n"
        "24. When answering facility locations, prioritize practical instructions over brand names.\n"
        "25. Never provide customs/GST information unless the user asks about declarations or tax."
    )

    messages = [("system", system_prefix)]
    for ex in examples:
        messages.append(("user", f"{ex['input']}\n\nContext:\n{ex['context']}"))
        messages.append(("ai", ex['answer']))
    messages.append(("user", "Question: {input}\n\nContext:\n{context}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    return create_stuff_documents_chain(llm=llm, prompt=prompt)

# 🔧 STEP 4: Main
def main():
    print("👋 Welcome to Changi Airport Chatbot!")
    print("🔍 Ask any question about Changi or Jewel. Type 'exit' to leave.")

    vectorstore = get_vectorstore()
    llm = get_llm()
    combiner = get_combiner(llm)

    session_id = "user-session-001"
    chat_history = InMemoryChatMessageHistory()

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue

        lower_q = question.lower()
        if lower_q in ["exit", "quit"]:
            print("🙏 Thank you for chatting. Have a safe journey! ✈️")
            break
        elif any(word in lower_q for word in ["thanks", "thank you", "appreciate", "cool", "nice", "great"]):
            print("🤖 You're most welcome! I'm always here to help ✨")
            continue

        metadata_filter = {}
        if "jewel" in lower_q: metadata_filter["location"] = "Jewel"
        elif "terminal 1" in lower_q: metadata_filter["location"] = "Terminal 1"
        elif "terminal 2" in lower_q: metadata_filter["location"] = "Terminal 2"
        elif "terminal 3" in lower_q: metadata_filter["location"] = "Terminal 3"
        elif "terminal 4" in lower_q: metadata_filter["location"] = "Terminal 4"

        if any(w in lower_q for w in ["food", "eat", "restaurant", "dining"]):
            metadata_filter["category"] = "dining"
        elif any(w in lower_q for w in ["shop", "shopping", "retail", "store"]):
            metadata_filter["category"] = "shopping"
        elif any(w in lower_q for w in ["transport", "taxi", "bus", "skytrain"]):
            metadata_filter["category"] = "transport"
        elif any(w in lower_q for w in ["attraction", "see", "visit", "explore", "fun"]):
            metadata_filter["category"] = "attractions"

        if "skytrain" in lower_q: metadata_filter["facility"] = "skytrain"
        elif "luggage" in lower_q: metadata_filter["facility"] = "luggage service"
        elif "taxi" in lower_q: metadata_filter["facility"] = "taxi"
        elif any(w in lower_q for w in ["toilet", "washroom", "restroom"]): metadata_filter["facility"] = "washroom"
        elif "security" in lower_q: metadata_filter["facility"] = "security"
        elif "customs" in lower_q: metadata_filter["facility"] = "customs"
        elif "lounge" in lower_q: metadata_filter["facility"] = "lounge"
        elif "shower" in lower_q or "relax" in lower_q: metadata_filter["facility"] = "shower"

        if "transit" in lower_q: metadata_filter["audience"] = "transit passengers"
        elif "family" in lower_q: metadata_filter["audience"] = "families"
        elif "business" in lower_q: metadata_filter["audience"] = "business travelers"

        # retriever = vectorstore.as_retriever(
        #     search_kwargs={
        #         "k": 3,
        #         "filter": {"$and": [{"key": k, "value": v} for k, v in metadata_filter.items()]} if metadata_filter else None,
        #         "score_threshold": 0.2
        #     }
        # )

        retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.2,
            "filter": metadata_filter or {}
        }
    )


        docs = retriever.invoke(question)

        if DEBUG:
            print("\n🧠 Chat History:")
            for m in chat_history.messages:
                print(f"[{m.type.upper()}] {m.content}")
            print("\n📄 Retrieved Chunks:")
            for doc in docs:
                print(doc.page_content[:300])

        if not docs:
            print("\n📌 I'm sorry! There is no provided context for the question you asked.\n")
            continue

        rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combiner)
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        # ✅ Inject prior messages for follow-up understanding
        prev_messages = ""
        for m in chat_history.messages[-4:]:
            if m.type == "human":
                prev_messages += f"User: {m.content}\n"
            elif m.type == "ai":
                prev_messages += f"AI: {m.content}\n"

        fused_input = f"{prev_messages}User: {question}"

        print("\n📌 Answer:", end=" ", flush=True)
        for chunk in chain_with_history.stream(
            {"input": fused_input},
            config=RunnableConfig(configurable={"session_id": session_id})
        ):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
            elif "output" in chunk:
                print(chunk["output"], end="", flush=True)

        print("\n")

if __name__ == "__main__":
    main()
