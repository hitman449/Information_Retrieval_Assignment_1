# IR Assignment 1 — Streamlit Information Retrieval System
**AIMLCZG537/DSECLZG537 | 2025-26 S2**

## Overview
End-to-end Information Retrieval system built with Streamlit, covering:
- **Section A**: Document collection (10 Wikipedia excerpts)
- **Section B**: Text preprocessing pipeline + Stemming vs Lemmatization comparison
- **Section C**: Phrase query using Biword Index & Positional Index
- **Section D**: Dictionary search using BST and B-Tree with benchmarks
- **Section E**: Tolerant retrieval (Wildcard/K-gram, Edit Distance, Soundex)
- **Section G**: Inferences & Discussion

## Files
```
ir_app/
├── app.py                  # Main Streamlit application
├── wikipedia_excerpts.py   # Dataset (10 Wikipedia documents)
└── README.md               # This file
```

## Install Dependencies
```bash
pip install streamlit nltk scikit-learn sortedcontainers pandas
```

Download NLTK data (auto-downloaded on first run, or manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```

## Run the App
```bash
streamlit run app.py
```

## Dataset
10 Wikipedia excerpts covering AI/CS topics:
- Artificial Intelligence, Machine Learning, NLP, Information Retrieval
- Search Engines, Deep Learning, Data Mining, Neural Networks
- Computer Vision, Knowledge Representation

You can also upload your own `.txt` files via the sidebar.

## Features by Section

### B. Text Preprocessing
- Tokenization, lowercasing, hyphen handling, stop word removal
- Stemming (Porter) and Lemmatization (WordNet)
- Side-by-side comparison with vocabulary size, TTR, and token-level examples

### C. Phrase Query
- Biword index and Positional index built at startup
- Real-time phrase search with result comparison
- False positive analysis and positional postings display

### D. Dictionary Search
- BST: simple binary search tree
- B-Tree: order-3 B-Tree implementation
- Benchmark table: time (µs) and comparison count per query
- Aggregate statistics and inference

### E. Tolerant Retrieval
- **Wildcard (K-gram)**: 2-gram index, pattern matching with regex verification
- **Edit Distance**: Levenshtein distance spell correction
- **Soundex**: Phonetic encoding for acoustically similar terms
