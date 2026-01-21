# 📄 Hybrid RAG: PDF + Wikipedia Question Answering

A **Hybrid Retrieval-Augmented Generation (RAG)** application that answers user questions by **prioritizing uploaded PDF documents** and **intelligently falling back to Wikipedia** when the document does not contain the required information.

Built using **LangChain (LCEL)**, **Groq LLM**, **FAISS**, **HuggingFace embeddings**, and an **interactive Streamlit UI**.

---

## 🚀 Features

- 📂 Upload a **PDF document**
- ❓ Ask questions in natural language
- 📄 Answers are grounded in the **PDF when relevant**
- 🌐 Automatic **Wikipedia fallback** when PDF lacks information
- 🏷️ Clearly shows **answer source** (PDF or Wikipedia)
- 🎨 Clean, interactive Streamlit interface
- 🔒 No hallucination — answers are context-bound

---

## 🧠 Why Hybrid RAG?

Document-only RAG systems fail when:
- The document doesn’t contain the answer
- The question is generic or conceptual

This project solves that by:
1. Searching the uploaded **PDF first**
2. Falling back to **Wikipedia only when required**

This ensures answers are **accurate, grounded, and complete**.

---

## 🏗️ Architecture

```text
User Question
      ↓
PDF Vector Search (FAISS)
      ↓
Relevant Context?
   ├── Yes → Answer from PDF
   └── No  → Wikipedia Search → Answer



<img width="887" height="682" alt="Screenshot 2026-01-21 163744" src="https://github.com/user-attachments/assets/915ceb4a-779b-44c2-a9e5-9ca89971e6b3" />

