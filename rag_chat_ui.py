import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from rag_chatbot import get_vectorstore, get_llm, get_combiner

# Load RAG components
@st.cache_resource
def load_chatbot():
    vectorstore = get_vectorstore()
    llm = get_llm()
    combiner = get_combiner(llm)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combiner)
    memory = InMemoryChatMessageHistory()
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history"
    ), memory, retriever

# Set up UI
st.set_page_config(page_title="Changi Airport Chatbot", page_icon="🛫")
st.title("🛫 Changi Airport AI Assistant")
st.markdown("Ask me anything about **Changi Airport** or **Jewel Changi** ✨")

(chatbot, chat_history, retriever) = load_chatbot()
session_id = "streamlit-session-001"

# Load or initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
user_input = st.chat_input("Ask your airport question here...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    retrieved_docs = []
    full_reply = ""

    with st.spinner("Thinking..."):
        try:
            # 🔍 Pre-fetch retrieved docs for debug
            retrieved_docs = retriever.invoke(user_input)

            # 🤖 Stream the reply
            with st.chat_message("ai"):
                response_box = st.empty()
                for chunk in chatbot.stream(
                    {"input": user_input},
                    config=RunnableConfig(configurable={"session_id": session_id})
                ):
                    token = chunk.get("answer") or chunk.get("output") or ""
                    full_reply += token
                    response_box.markdown(full_reply)

        except Exception as e:
            error_msg = f"❌ Error generating response: {e}"
            st.error(error_msg)
            full_reply = error_msg

    # 💾 Store AI response
    if full_reply and not full_reply.startswith("❌"):
        st.session_state.messages.append({"role": "ai", "content": full_reply})
        st.rerun()
