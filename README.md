📘 University Student Complaints Topic Modeling

LSA • LDA • BERTopic

This repository contains a full pipeline for extracting and analyzing topics from a dataset of student complaints.
The chosen dataset is University Students Complaints & Reports from Kaggle: https://www.kaggle.com/datasets/omarsobhy14/university-students-complaints-and-reports
The dataset contains 1005 records of complaints. The goal of the project is to use NLP techniques to produce a list of most frequently addressed topics.

The project implements classical and modern topic modeling approaches, provides visualizations, and includes reusable components for preprocessing, vectorization, and modeling.

## 📑 Table of Contents

- [Project Pipeline](#project-pipeline)
- [Repository Structure](#repository-structure)
- [Topic Modeling Methods](#topic-modeling-methods)
- [How to Run](#how-to-run)
- [License](#license)

## Project Pipeline

**_The LSA/LDA pipeline includes:_**

Text cleaning & lemmatization (nltk, spaCy, re)

Bi-grams (gensim)

Vectorization (TF-IDF / BOW)

LSA and LDA (gensim.models)

Visual analytics (wordcloud) 

  
**_The BERTopic pipeline includes:_**

Text cleaning & lemmatization (nltk, spaCy, re)

BERTopic (bertopic)

Visual analytics (table)


## Repository Structure
<pre>project/
│
├── data/                  # Raw dataset
│   └── Datasetprojpowerbi.csv
│
├── notebooks/             # Jupyter notebooks with each model experiments
│   └── bertopic_analysis.ipynb
│   └── lda_analysis.ipynb
│   └── lsa_analysis.ipynb
│
├── src/
│   ├── metrics/
│   |   └── wetc.py        # WETC calculation
│   |
│   ├── preprocessing/     # Cleaning, lemmatization, stopwords
|   |   └── clean_text.py  
│   │
│   ├── vectorization/     # TF-IDF, BoW, Vocabulary
│   │   └── vecorize_lsa_lda.py
│   │
│   └── topic_models/      # LSA, LDA, BERTopic training & saving
│       └── lsa_lda_model.py
│       └── bertopic_model.py
│
├── requirements.txt
└── README.md</pre>

## Topic Modeling Methods
| Method   | Vectorization                | Main process                              | Method class    | Speed   |  Quality of Topics
|----------|------------------------------|-------------------------------------------|-----------------|---------|--------------------
| LSA      | TF-IDF                       | SVD for the corpus matrix                 | Classic ML      | Fast    |  Medium
| LDA      | BoW                          | Iterative topic probablity approximation  | Classic ML      | Medium  |  Medium
| BERTopic | SBERT sentence embeddings    | HDBSCAN docs culstering                   | Deep Learning   | Slow    |  High

## How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run notebook with the chosen model

## License

MIT License.
Feel free to use and adapt the code.
