# 📄 Conversational RAG with PDF Uploads

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload one or more PDFs and have a natural, multi-turn conversation about their content — complete with chat history awareness, so follow-up questions like *"summarize our conversation"* actually work.

Built with **LangChain**, **Groq (Llama 3.3 70B)**, **ChromaDB**, and **HuggingFace Embeddings**, wrapped in a clean **Streamlit** chat interface.

---

## ✨ Features

- 🔑 **Bring your own Groq API key** — entered securely at runtime, never hardcoded
- 📚 **Multi-PDF upload** — chat across the combined content of several documents at once
- 🧠 **History-aware retrieval** — reformulates follow-up questions using chat context before retrieving (e.g. "what about its formula?" resolves to "what is the formula for attention?")
- 💬 **Session-based memory** — each `Session ID` maintains its own independent chat history, so multiple users/conversations don't mix
- 🎨 **Modern chat UI** — real chat bubbles via `st.chat_message`, sidebar settings, loading spinners, and a debug panel tucked out of the way
- 🗑️ **One-click history reset** — clear a session's memory without restarting the app

---

## 🖼️ Demo

| Upload & Ask | Chat Flow |
<img width="1482" height="587" alt="Screenshot 2026-07-19 173340" src="https://github.com/user-attachments/assets/9fa5873d-7ff3-4b83-9d95-e1994d7cae49" />


> Example: after asking *"What are transformers in detail?"*, a follow-up like *"What is attention and can you give the formula?"* is understood in context — and the bot can even summarize the whole conversation on request.

---

## 🏗️ How It Works

```
PDF Upload
   │
   ▼
PyPDFLoader → Text Splitting (RecursiveCharacterTextSplitter)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) → ChromaDB Vector Store
   │
   ▼
User Question ──► History-Aware Retriever ──► Reformulated Standalone Question
                          │
                          ▼
                 Relevant Chunks Retrieved
                          │
                          ▼
              Groq LLM (Llama 3.3 70B) + Chat History
                          │
                          ▼
                     Final Answer
```

**Key design pieces:**

- **`create_history_aware_retriever`** — before hitting the vector store, the latest question is rewritten into a context-free, standalone question using the chat history. This is what lets pronouns and implicit references ("it", "that", "the formula") resolve correctly.
- **`create_stuff_documents_chain`** — "stuffs" all retrieved chunks directly into the prompt context for the answer-generation step.
- **`RunnableWithMessageHistory`** — wraps the whole RAG chain so LangChain automatically injects and updates chat history per `session_id`, backed by an in-memory `ChatMessageHistory` store.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B Versatile |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| PDF Parsing | PyPDFLoader |
| UI | Streamlit |
| Env Management | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies
```bash
pip install streamlit langchain langchain-groq langchain-chroma langchain-huggingface langchain-community langchain-text-splitters python-dotenv pypdf
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
HF_TOKEN=your_huggingface_token_here
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use it
1. Enter your **Groq API key** in the sidebar ([get one free here](https://console.groq.com/keys))
2. Upload one or more PDF files
3. Start chatting in the input box at the bottom — ask questions, follow up naturally, or ask for a summary of the conversation

---

## 📌 Notes & Limitations

- The vector store is rebuilt in-memory on every upload (not persisted to disk), so re-running the app clears the index.
- Chat history is stored per `session_id` in Streamlit's session state — it resets if the app restarts or the browser session ends.
- Answers are intentionally kept concise (max 3 sentences) via the system prompt, tunable in `app.py`.

---

## 👤 About the Author

**Prithvi Raj Mukhiya**
🎓 IIIT Kota — Electronics & Communication Engineering
💡 Interested in **Machine Learning, Deep Learning, NLP, and GenAI applications**

Feel free to connect, share feedback, or suggest improvements!

---

## 📄 License

This project is open for learning and personal use. Feel free to fork and build on it.
