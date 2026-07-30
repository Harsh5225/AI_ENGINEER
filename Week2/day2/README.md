---

# 📚 Automated QA & Test-Prep Generator using RAG and LangChain

An automated pipeline built with **LangChain**, **Groq (Llama-3.3-70B)**, and **FAISS** that ingests documentation/PDFs, automatically generates comprehensive practice questions, retrieves context using vector similarity search, and answers those questions to produce a complete test-prep study guide.

---

## 🏗️ Architecture & Workflow

```
[ PDF Document ] ──► PyPDFLoader
                          │
                          ├─────────────────────────────────────────┐
                          ▼                                         ▼
               Large Token Splitting                     Small Text Splitting
                   (10k Tokens)                              (1k Tokens)
                          │                                         │
                          ▼                                         ▼
                 Summarize Chain                        HuggingFace Embeddings
                 (Refine Method)                                    │
                          │                                         ▼
                          ▼                                 FAISS VectorStore
                 Generated Questions                                │
                          │                                         │
                          └─────────────► RetrievalQA ◄─────────────┘
                                              │
                                              ▼
                                       [ answers.txt ]

```

1. **Document Ingestion:** PDF files are extracted into LangChain `Document` objects.
2. **Question Generation Chain:**
* Text is chunked into larger token windows to retain high-level context.
* LangChain's `load_summarize_chain` with a **Refine** strategy sequentially analyzes the text to build and iterate on exam-ready questions.


3. **Chunking & Vector Embeddings:**
* Text is re-chunked into smaller segments (ideal for precise retrieval).
* Dense embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` and stored in a **FAISS** vector database.


4. **Automated Retrieval QA:**
* Questions are extracted using regular expressions.
* A **RetrievalQA** pipeline fetches top matching context from FAISS and prompts **Llama 3.3 70B (via Groq)** to generate answers.
* Results are saved automatically to `answers.txt`.



---

## 🧰 Tech Stack

* **Orchestration Framework:** LangChain (`langchain-core`, `langchain-community`, `langchain-classic`)
* **LLM Provider:** Groq API (`ChatGroq` using `llama-3.3-70b-versatile`)
* **Embeddings:** HuggingFace Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Document Loader:** `PyPDFLoader`

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Environment Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/qa-generator-rag.git
cd qa-generator-rag

pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu pypdf python-dotenv

```

### 3. API Key Configuration

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here

```

### 4. Input Preparation

Place your PDF document in the designated directory:

```
data/SDG.pdf

```

### 5. Execution

Run the main pipeline script:

```bash
python main.py

```

The output questions and corresponding answers will be written to `answers.txt`.

---

## ⚙️ Key Technical Concepts Explained

### TokenTextSplitter vs. RecursiveCharacterTextSplitter

In this project, chunking strategy selection impacts performance:

| Feature | `TokenTextSplitter` | `RecursiveCharacterTextSplitter` |
| --- | --- | --- |
| **Splitting Mechanism** | Splits strictly based on BPE/LLM token counts. | Recursively splits by characters (`\n\n`, `\n`, ` `, `""`). |
| **Context Boundaries** | Hard token boundaries (may split mid-sentence). | Tries to preserve paragraphs and complete sentences. |
| **Primary Advantage** | Strict context window boundary management. | Preserves semantic coherence for better vector retrieval. |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
