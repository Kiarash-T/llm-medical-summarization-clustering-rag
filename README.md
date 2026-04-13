# Cluster-Aware Medical Report Summarization

This repository contains the implementation of two research works focused on improving medical report summarization using Large Language Models (LLMs), patient clustering, and retrieval-augmented generation techniques.

## 📄 Included Papers

### 1. *Embedding-Enhanced Patient Clustering for Customized Medical Report Summarization using LLMs*

This work introduces a pipeline that combines clinical data embeddings and clustering methods (e.g., t-SNE, UMAP, t-SNE-PSO) with LLM-based summarization. Patients are grouped into clinically meaningful clusters, and summaries are generated at the cluster level to improve contextual relevance and personalization.

### 2. *Cluster-Aware Retrieval-Augmented Generation with Hybrid Retrieval for Faithful Medical Report Summarization*

This work extends the previous approach by incorporating a cluster-aware Retrieval-Augmented Generation (RAG) framework. It introduces hybrid retrieval (TF-IDF + dense embeddings) and evidence-grounded generation to improve factual accuracy, reduce hallucinations, and enforce citation-based summarization.

## 🧠 Repository Structure

* `clustering-llm-baseline/`
  Implementation of embedding-based patient clustering and cluster-wise LLM summarization (Paper 1).

* `cluster-aware-rag/`
  Implementation of the cluster-aware RAG framework with hybrid retrieval and evidence-constrained generation (Paper 2).

## 🎯 Objective

The goal of this repository is to explore and improve:

* Context-aware medical summarization through patient clustering
* Faithfulness and reliability of LLM-generated summaries
* Reduction of hallucinations using retrieval-based grounding
* Scalable and clinically meaningful summarization pipelines

## ⚠️ Note

The implementations are based on synthetic datasets due to limited access to real clinical data and are intended for research and experimentation purposes only.

## 📚 Citation

If you use this repository, please cite the corresponding papers listed above.

