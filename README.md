
# 🤖 AI_ENGINEER

### Building • Learning • Experimenting with Agentic AI & Generative AI

> A hands-on journey into **LLMs, RAG, LangChain, LangGraph, AI Agents, Tool Calling, Memory, HITL, Observability, and Production AI Systems.**

This repository documents my journey from understanding **LLM fundamentals** to building and deploying **stateful, tool-using Agentic AI applications**.

Rather than only learning concepts theoretically, I focus on:

**Learn → Understand → Implement → Experiment → Deploy → Document**

---

## 🚀 What I'm Building

This repository contains hands-on implementations, experiments, mini-projects and production-oriented AI applications covering:

- 🧠 LLM Applications
- 🔗 LangChain & LCEL
- 🕸️ LangGraph
- 📚 Retrieval-Augmented Generation (RAG)
- 🛠️ Tool Calling & Function Calling
- 🧩 Agentic Workflows
- 💾 Short-term & Long-term Memory
- 👨‍💻 Human-in-the-Loop (HITL)
- 🔀 Conditional & Parallel Workflows
- 🔁 Evaluator-Optimizer Workflows
- 🔎 MMR Retrieval
- 📊 LangSmith Observability
- 💽 Persistent State & Checkpointing
- 🎨 Streamlit AI Interfaces
- ☁️ AI Application Deployment

---

# ⭐ Featured Project

## 🤖 Agentic Chatbot

A stateful Agentic AI chatbot built with **LangGraph**, capable of dynamically using external tools and maintaining persistent conversations.

### Architecture

```text
                    User
                     │
                     ▼
              ┌──────────────┐
              │  LangGraph   │
              │     Agent    │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │     LLM      │
              │ Tool Decision│
              └──────┬───────┘
                     │
              Tool Required?
                ┌────┴────┐
               YES        NO
                │          │
                ▼          ▼
        ┌──────────────┐  Response
        │  Tool Node   │
        └──────┬───────┘
               │
        ┌──────┼───────────────┐
        ▼      ▼       ▼       ▼
      Search  Math   Weather  Stocks
        │      │       │       │
        └──────┴───────┴───────┘
               │
               ▼
              LLM
               │
               ▼
          Final Response
````

### Key Features

* 🔀 Dynamic tool selection
* 🔍 Web search
* 🧮 Calculator
* 🌤️ Real-time weather
* 📈 Stock information
* 📚 RAG-based knowledge retrieval
* 💬 Persistent conversation memory
* 🧵 Thread-based conversations
* 🛑 Human-in-the-Loop workflows
* 🔎 LangSmith observability
* ⚡ Streaming responses
* 🎨 Streamlit interface
* ☁️ Deployed application

### Agentic Loop

```text
User Input
    ↓
LLM
    ↓
Reason / Decide
    ↓
Tool Call
    ↓
Tool Result
    ↓
LLM
    ↓
Final Response
```

The goal is to move beyond:

```text
Prompt → LLM → Response
```

towards:

```text
Understand → Decide → Act → Observe → Decide → Respond
```

---

# 🧠 Agentic AI Concepts Implemented

## 1. LLM Fundamentals

Understanding the foundation behind modern AI applications:

* Prompting
* System / Human / AI messages
* Temperature
* Streaming
* Model providers
* Structured outputs
* Function / Tool calling

---

## 2. LangChain

Hands-on work with:

* Chat Models
* Prompt Templates
* Output Parsers
* Chains
* LCEL
* Runnables
* Retrievers
* Vector Stores
* Memory

### LCEL

```text
Input
  ↓
Prompt
  ↓
LLM
  ↓
Output Parser
  ↓
Response
```

---

# 📚 RAG — Retrieval-Augmented Generation

Implemented end-to-end RAG pipelines:

```text
Documents
    ↓
Load
    ↓
Split
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
Relevant Context
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```

### Technologies

* FAISS
* HuggingFace Embeddings
* LangChain Retrievers
* PDF Loaders
* MMR Retrieval

### MMR

Explored **Maximal Marginal Relevance** to balance:

```text
Relevance + Diversity
```

instead of retrieving multiple highly similar chunks.

---

# 🕸️ LangGraph

LangGraph is the core framework used in this repository for building stateful and controllable AI workflows.

### Core Concepts

* State
* Nodes
* Edges
* Conditional Edges
* Reducers
* Checkpointers
* Threads
* Interrupts
* Tool Nodes
* Routing
* Loops

### Basic Mental Model

```text
             State
               │
               ▼
             Node
               │
        ┌──────┴──────┐
        ▼             ▼
     Node A         Node B
        │             │
        └──────┬──────┘
               ▼
             Node C
```

---

# 🔀 Agentic Workflow Patterns

Implemented and studied several important workflow patterns.

### Conditional Workflow

```text
Input
 ↓
Decision
 ├── Path A
 └── Path B
```

### Parallel Workflow

```text
             ┌──► Task A ──┐
Input ───────┼──► Task B ──┼──► Aggregator
             └──► Task C ──┘
```

### Evaluator → Optimizer

```text
Generate
   ↓
Evaluate
   ↓
Approved? ───── YES ───► END
   │
   NO
   ↓
Optimize
   ↓
Evaluate Again
```

These patterns form the foundation for more complex agentic systems.

---

# 🛑 Human-in-the-Loop

Explored HITL workflows where an AI agent should **not execute sensitive actions autonomously**.

Example:

```text
User
 ↓
Agent
 ↓
Analyze Request
 ↓
Stock Purchase Required
 ↓
🛑 HUMAN APPROVAL
 ↓
 ┌─────────────┐
 │             │
Approve       Reject
 │             │
 ▼             ▼
Execute       Stop
```

This demonstrates an important principle of production AI:

> **Autonomy should exist within controlled boundaries.**

---

# 💾 Agent Memory

Explored multiple levels of memory:

### Short-Term Memory

Maintaining conversation state using:

* State
* Reducers
* `add_messages`

### Persistent Memory

Using:

* SQLite
* LangGraph Checkpointing
* Thread IDs

```text
Thread ID
    ↓
Checkpoint
    ↓
Conversation State
    ↓
Persistent Storage
```

The chatbot can therefore maintain separate conversations and recover them after application restarts.

---

# 🔎 Observability

Integrated **LangSmith** to understand what happens inside an AI application.

Monitoring includes:

* LLM calls
* Node execution
* Tool calls
* Latency
* Token usage
* Errors
* Workflow traces

This makes debugging complex agentic workflows much easier.

---

# 🛠️ Technology Stack

### AI / GenAI

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-FF6B35?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)

### Retrieval & Memory

![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge\&logo=sqlite\&logoColor=white)

* HuggingFace Embeddings
* RAG
* MMR
* Checkpointing

### Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)

* Streamlit
* REST APIs
* External Tools
* API Integration

### Observability & Deployment

* LangSmith
* Render
* Environment Variables
* Git & GitHub

---

# 📂 Repository Structure

```text
AI_ENGINEER/
│
├── 🤖 Agentic_chatbot/
│   ├── RAG
│   ├── Tool Calling
│   ├── LangGraph
│   ├── Memory
│   ├── HITL
│   ├── LangSmith
│   ├── Streamlit
│   └── Deployment
│
├── 🕸️ LangGraphLearn/
│   ├── Conditional Workflows
│   ├── Parallel Workflows
│   ├── Evaluator-Optimizer
│   ├── Memory
│   ├── Checkpointing
│   └── Agent Workflows
│
├── 🔗 Week1/
│   ├── LangChain Fundamentals
│   ├── LCEL
│   ├── Runnables
│   ├── Prompt Templates
│   └── Output Parsers
│
├── 🧠 Week2/
│   ├── RAG
│   ├── Retrieval
│   ├── Memory
│   └── Advanced LangChain Concepts
│
└── README.md
```

---

# 🧪 Hands-On Experiments

Some of the concepts implemented in this repository include:

| Concept                | Implementation |
| ---------------------- | -------------- |
| LLM Calls              | ✅              |
| Streaming              | ✅              |
| Prompt Templates       | ✅              |
| LCEL                   | ✅              |
| Runnables              | ✅              |
| Output Parsers         | ✅              |
| RAG                    | ✅              |
| FAISS                  | ✅              |
| HuggingFace Embeddings | ✅              |
| MMR                    | ✅              |
| LangGraph              | ✅              |
| Conditional Routing    | ✅              |
| Parallel Workflows     | ✅              |
| Reducers               | ✅              |
| Evaluator-Optimizer    | ✅              |
| Memory                 | ✅              |
| SQLite Persistence     | ✅              |
| Tool Calling           | ✅              |
| Web Search             | ✅              |
| Weather API            | ✅              |
| Stock API              | ✅              |
| Human-in-the-Loop      | ✅              |
| LangSmith              | ✅              |
| Streamlit              | ✅              |
| Deployment             | ✅              |

---

# 🎯 Current Focus

My current focus is moving from **individual AI concepts** towards complete **production-oriented Agentic AI systems**.

### Current Learning Path

```text
LLMs
 ↓
LangChain
 ↓
LCEL
 ↓
RAG
 ↓
Memory
 ↓
LangGraph
 ↓
Workflows
 ↓
Tool Calling
 ↓
HITL
 ↓
Persistence
 ↓
Observability
 ↓
Multi-Agent Systems
 ↓
Production AI
```

---

# 🚀 What's Next?

I'm currently exploring:

* [ ] Advanced Agentic Workflows
* [ ] Multi-Agent Systems
* [ ] Agent Planning
* [ ] Better RAG Pipelines
* [ ] RAG Evaluation
* [ ] Agent Evaluation
* [ ] Long-Term Memory
* [ ] Production-grade APIs
* [ ] Authentication & Authorization
* [ ] Dockerization
* [ ] CI/CD for AI Applications
* [ ] Scalable AI Systems
* [ ] End-to-End Agentic AI Projects

---

# 💡 Philosophy

I believe the best way to learn AI engineering is not just to consume tutorials.

It is to **build systems**.

Every concept in this repository follows the same cycle:

```text
📖 Learn
   ↓
🧠 Understand
   ↓
💻 Implement
   ↓
🧪 Experiment
   ↓
🐛 Debug
   ↓
🚀 Deploy
   ↓
📝 Document
```

This repository is my public learning lab for becoming a better **AI Engineer**.

---

# 📊 Learning in Public

I am consistently documenting my AI/Agentic AI learning journey and sharing what I build, the problems I encounter, and the concepts I understand.

> **Learning → Building → Sharing → Improving**

---

# 👨‍💻 About Me

I'm a 2026 ECE graduate focused on building a career in **Software Engineering, Generative AI and Agentic AI**.

My background combines:

* Data Structures & Algorithms
* Backend Development
* Java / Spring Boot
* Python
* Machine Learning
* Deep Learning
* Generative AI
* Agentic AI
* System Design

### Current Goal

> Build AI systems that don't just generate text — but can **reason, retrieve, use tools, maintain state, take controlled actions, and adapt to changing situations.**

---

⭐ If you find this repository useful, consider giving it a star.

**More experiments. More systems. More learning. 🚀**
