# Secure Enterprise RAG Assistant

## Overview

This project started as a learning exercise to better understand how Retrieval-Augmented Generation (RAG) systems work beyond simple chatbot demos.

I wanted to build the core components myself and experiment with different retrieval techniques, access control mechanisms, evaluation methods, and document ingestion pipelines.

The project simulates an enterprise knowledge assistant that can answer questions from a collection of internal documents while respecting role-based access restrictions.

---

## What I Wanted To Learn

While building this project, I explored:

* Document ingestion and preprocessing
* Chunking strategies
* Hybrid retrieval (keyword + semantic search)
* Vector databases and embeddings
* Reranking techniques
* Grounded answer generation
* Citation generation
* Role-Based Access Control (RBAC)
* RAG evaluation metrics
* Enterprise-style document organization

The focus was understanding how the individual pieces of a RAG pipeline work together rather than building a production-ready application.

---

## High-Level Architecture

User Query
    ↓
    
Query Planning
    ↓
    
Retrieval
    ├─ Keyword Search
    ├─ Semantic Search
    └─ Vector Search
    ↓
    
Reranking
    ↓
    
Context Assembly
    ↓
    
Answer Generation
    ↓
    
Response + Citations
---

## Features

* Multi-format document ingestion

  * PDF
  * Markdown
  * JSON
  * CSV
  * SQL exports

* Hybrid retrieval pipeline
* Role-based document access
* Citation-aware responses
* Optional LLM-backed generation
* Retrieval diagnostics
* Evaluation framework
* Synthetic enterprise-style knowledge base
---

## Tech Stack

* Python
* Streamlit
* ChromaDB
* Ollama (optional)
* OpenAI-compatible APIs (optional)
* PyPDF
* Pytest

---
## Running The Project

Create a virtual environment: python -m venv .venv

Activate it: .venv\Scripts\activate

Install dependencies: pip install -r requirements.txt

Run the application: streamlit run app.py

The application will be available at: http://localhost:8501

---

## Evaluation

The project includes a small evaluation framework for measuring retrieval quality.

Current metrics include:

* Recall@K
* MRR (Mean Reciprocal Rank)
* Citation Accuracy
* Faithfulness
* Context Precision
---

## Dataset

The repository contains a synthetic enterprise-style dataset created for experimentation and learning purposes.

The data includes:

* HR policies
* Security documentation
* IT procedures
* Finance reports
* Project documentation
* Audit records

No real company or customer data is used.(I did not get any :/ )
---
------------------------------------THE END------------------------------------------------
