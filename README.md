# MemoryOS – Your Personal AI Memory Layer

> **Don’t remember where you saved it. Just remember what you need.**

## 📌 Overview

MemoryOS is a personal AI memory layer designed to bring fragmented digital information into one searchable memory system.

Instead of remembering **where** a particular piece of information was saved, users can upload their documents and ask questions in natural language.

MemoryOS processes the uploaded information, converts it into searchable vector representations, retrieves relevant memories, and uses a Retrieval-Augmented Generation (RAG) pipeline to provide an AI-generated answer along with supporting sources.

---

## 🎯 Problem Statement

Important personal information is often scattered across:

- PDF documents
- Notes
- Text files
- Emails
- Meeting records
- Cloud files
- Other digital sources

Finding a specific piece of information requires remembering where it was stored and manually searching through multiple files.

### The problem

> **We remember the information we need, but often forget where we stored it.**

MemoryOS aims to solve this problem by creating a unified AI-powered memory layer.

---

## 💡 Proposed Solution

MemoryOS allows users to upload documents and interact with their stored information using natural-language questions.

The system:

1. Accepts a document from the user.
2. Extracts its text.
3. Splits the text into smaller chunks.
4. Converts the chunks into embeddings.
5. Stores the embeddings in a vector database.
6. Converts the user's question into an embedding.
7. Searches for the most relevant stored information.
8. Sends the retrieved context to an LLM through a RAG pipeline.
9. Generates an answer based on the retrieved information.
10. Returns the answer along with source information.

---

## ✨ Current Features

The current implementation includes:

- 📄 Document upload
- 📑 PDF/text extraction
- ✂️ Text chunking
- 🧠 Sentence Transformer embeddings
- 🔎 Semantic similarity search
- 🗄️ ChromaDB vector storage
- 💬 Ask Your Memory interface
- 🤖 Retrieval-Augmented Generation (RAG) pipeline
- 📚 Retrieved source/evidence display
- ⚡ FastAPI backend
- ⚛️ React frontend
- 🌙 Modern dark-themed dashboard

---

## 🔄 System Workflow

```text
User Uploads Document
        ↓
   File Validation
        ↓
   Text Extraction
        ↓
      Chunking
        ↓
    Embeddings
        ↓
     ChromaDB
        ↓
   Vector Storage
        ↓
 User Asks Question
        ↓
  Query Embedding
        ↓
 Semantic Search
        ↓
 Retrieve Relevant Chunks
        ↓
     RAG Prompt
        ↓
      OpenAI LLM
        ↓
   AI Generated Answer
        ↓
 Answer + Sources
