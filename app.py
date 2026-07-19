## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## ---------------- Streamlit Page Setup ----------------
st.set_page_config(
    page_title="Conversational RAG with PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Conversational RAG With PDF Uploads")
st.caption("Upload PDFs and chat with their content — powered by LangChain + Groq")

## ---------------- Sidebar: Settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password")
    session_id = st.text_input("Session ID", value="default_session")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type="pdf", accept_multiple_files=True
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if "store" in st.session_state and session_id in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        st.rerun()

## Check if groq api key is provided
if not api_key:
    st.warning("Please enter the Groq API Key in the sidebar to get started.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

## statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to begin chatting.")
    st.stop()

## Process uploaded PDF's
with st.spinner("Reading and indexing your PDF(s)..."):
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

st.success(f"✅ Indexed {len(uploaded_files)} file(s): {', '.join(f.name for f in uploaded_files)}")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

## Answer question
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

## ---------------- Chat UI ----------------
session_history = get_session_history(session_id)

# Render past messages using chat bubbles
for msg in session_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input pinned to the bottom
user_input = st.chat_input("Ask a question about your PDF(s)...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
        st.markdown(response['answer'])

# Optional: view raw session state / debug info without cluttering the UI
with st.sidebar:
    with st.expander("🔍 Debug: Session Store"):
        st.write(st.session_state.store)