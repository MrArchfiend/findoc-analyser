# FinDoc: Financial Document Analysis using AI 
This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** pipeline for analyzing SEC and 10K financial documents. The project addresses key limitations of large language models (LLMs) in financial analysis, such as hallucination, limited context size, and difficulty in handling complex queries.
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation Results](#evaluation-results)
- [Contributors](#contributors)

---

## Overview
The project focuses on leveraging a multi-layer RAG pipeline for strategic financial investment analysis. By integrating LLMs with advanced retrieval techniques, the pipeline ensures accurate and reliable answers to both broad and foundational financial questions.

### Key Goals
1. **Overcome LLM limitations**: Handle hallucinations, complex parsing, and restricted context size.
2. **Sub-question generation**: Enhance query precision with sub-questions.
3. **Modular workflow**: Allow customizable and reusable components for financial document analysis.

---

## Features
- **Financial Document Processing**: Chunking SEC and 10K documents into manageable pieces for retrieval.
- **Chunking Methods**: 
  - Fixed-size Chunking
  - Sentence-based Chunking
- **Retrieval with ChromaDB**: Indexing and retrieving relevant document chunks based on cosine similarity.
- **Sub-Question Generation**: Break down broad queries into specific sub-questions.
- **Evaluation Framework**: Metrics for response relevancy, context precision, and faithfulness using ROUGE/deepeval.

---

## Installation
1. Install Ollama (local LLM runner):
  
2. Clone the repository:
   
3. Install dependencies:
   
4. Configure constants in `constants.py`:
   

> No API keys required. All components run fully locally for free.

---

## Usage
### Running the Streamlit App
Start Ollama, then launch the Streamlit interface to process documents and generate responses:


### Testing the Pipeline
Run the Jupyter notebook to evaluate performance:


---

## Project Structure
```
findoc-analyser/
├── chromadb/                   # ChromaDB database for indexing
├── data/                       # Directory for financial documents
├── env/                        # Environment configurations
├── testing/                    # Test scripts
├── app.py                      # Streamlit app
├── chunking.py                 # Chunking methods
├── constants.py                # Model config and constants (no API keys needed)
├── data_preprocessing.py       # Financial document preprocessing
├── evaluation_results_rag.csv  # Evaluation results
├── evaluation.py               # Evaluation metrics and scoring
├── question_generation.py      # Sub-question generation logic
├── response_generation.py      # Ollama-based response generation
├── storing_retrieval.py        # Storing and retrieving chunks with ChromaDB
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
└── test.ipynb                  # Evaluation and testing notebook
```

---

## Evaluation Results
The evaluation results are saved in `evaluation_results_rag.csv`

For more details, see the `test.ipynb` notebook.

---

## Contributors
- **Aditya Singh**: Architecture Design and Model Development.
- **Dheeraj Yadav**: Integration, UI/UX and Deployment
